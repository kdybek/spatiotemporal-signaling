import numpy as np
from tqdm import tqdm
from absl import app, flags
import optax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import random
import os
import wandb
import json

from utils.model import get_rvm
from utils.logging import get_exp_name, setup_wandb
from utils.dataloader import create_train_test_datasets, TransformPipeline, prepare_rvm_src_tgt_pairs, batch_iterator
from utils.evaluation import full_evaluation
from utils.loss import update_model


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('steps', 300_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50_000, 'Saving interval.')
flags.DEFINE_string('dataset_path', 'toy_dataset.zarr',
                    'Path to the train dataset.')
flags.DEFINE_string('save_dir', 'checkpoints',
                    'Directory to save model checkpoints.')

flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for the optimizer.')
flags.DEFINE_integer(
    'batch_size', 16, 'Batch size for training and evaluation.')
flags.DEFINE_float('train_split', 0.5,
                   'Proportion of data to use for training (rest is for validation).')
flags.DEFINE_integer('clip_size', 224, 'Height and width of input images.')
flags.DEFINE_integer('clip_frames', 16, 'Number of frames in each video clip.')
flags.DEFINE_integer(
    'acq_freq', 30, 'Acquisition frequency (in minutes) for sampling video clips.')
flags.DEFINE_string('channel_names', 'Ch_ERK-KTR',
                    'Space-separated list of channel names to use from the videos.')

flags.DEFINE_integer('src_frames', 4, 'Number of source frames for reconstruction.')
flags.DEFINE_integer('tgt_frames', 4, 'Number of target frames for reconstruction.')
flags.DEFINE_float('masking_ratio', 0.95,
                   'Ratio of target tokens to mask during training.')
flags.DEFINE_string('encoder_variant', 'L', 'Variant of the encoder to use.')

flags.DEFINE_string('transforms', 'arcsinh butterworth percentile_norm',
                    'Space-separated list of transforms to apply to the videos. Supported: percentile_norm, arcsinh, log1p, butterworth.')
flags.DEFINE_float('arcsinh_cofactor', 5.0,
                   'Cofactor for arcsinh transform.')
flags.DEFINE_float('butterworth_cutoff', 0.2,
                   'Cutoff frequency for Butterworth low-pass filter.')
flags.DEFINE_integer('butterworth_order', 2,
                     'Order of Butterworth low-pass filter.')
flags.DEFINE_boolean('per_frame_butterworth', False,
                     'Whether to apply Butterworth filter independently to each frame (instead of across time).')

flags.DEFINE_string('checkpoint_path', None, 'Path to a checkpoint to load model parameters from.')


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)


def initialize_model(model, rng_key):
    num_channels = len(FLAGS.channel_names.split())
    dummy_src = jnp.ones((1, FLAGS.src_frames, FLAGS.clip_size,
                         FLAGS.clip_size, num_channels), dtype=jnp.float32)
    dummy_tgt = jnp.ones((1, FLAGS.tgt_frames, FLAGS.clip_size, FLAGS.clip_size,
                         num_channels), dtype=jnp.float32)
    dummy_deltas = jnp.zeros((1, FLAGS.tgt_frames), dtype=jnp.int32)

    params = model.init(rng_key, dummy_src, dummy_tgt, dummy_deltas)['params']
    return params


def create_optimizer(
    base_lr,
    warmup_steps,
    total_steps,
):
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )

    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
    )

    schedule = optax.join_schedules(
        schedules=[
            warmup_schedule,
            cosine_schedule,
        ],
        boundaries=[
            warmup_steps,
        ],
    )

    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=0.05,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
    )

    return optimizer


def load_checkpoint(checkpointer, checkpoint_path):
    state = checkpointer.restore(os.path.abspath(checkpoint_path))
    return state['params'], state['opt_state'], state['step']


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='Spaciotemporal Signaling',
                group=FLAGS.run_group, name=exp_name)
    set_seed(FLAGS.seed)
    rng_key = jax.random.PRNGKey(FLAGS.seed)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flags_dict = FLAGS.flag_values_dict()

    with open(os.path.join(FLAGS.save_dir, 'config.json'), 'w') as f:
        json.dump(flags_dict, f, indent=4)

    transform_pipeline = TransformPipeline(
        transform_names_list=FLAGS.transforms.split(),
        arcsinh_cofactor=FLAGS.arcsinh_cofactor,
        butterworth_cutoff=FLAGS.butterworth_cutoff,
        butterworth_order=FLAGS.butterworth_order,
        per_frame_butterworth=FLAGS.per_frame_butterworth,
    )

    train_dataset, test_dataset = create_train_test_datasets(
        test_fraction=1 - FLAGS.train_split,
        zarr_path=FLAGS.dataset_path,
        clip_frames=FLAGS.clip_frames,
        clip_size=FLAGS.clip_size,
        acq_freq=FLAGS.acq_freq,
        channel_names_list=FLAGS.channel_names.split(),
        transform_pipeline=transform_pipeline,
    )

    model = get_rvm(
        num_channels=len(FLAGS.channel_names.split()),
        masking_ratio=FLAGS.masking_ratio,
        encoder_variant=FLAGS.encoder_variant,
    )

    init_key, rng_key = jax.random.split(rng_key)
    params = initialize_model(model, init_key)

    optimizer = create_optimizer(
        base_lr=FLAGS.learning_rate,
        warmup_steps=int(0.05 * FLAGS.steps),
        total_steps=FLAGS.steps,
    )
    opt_state = optimizer.init(params)

    if FLAGS.checkpoint_path is not None:
        eval_key, rng_key = jax.random.split(rng_key)
        eval_metrics = full_evaluation(
            model, test_dataset, params, FLAGS.src_frames, FLAGS.tgt_frames, FLAGS.batch_size, eval_key
        )
        wandb.log(eval_metrics, step=0)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    @jax.jit
    def training_step(
        params,
        opt_state,
        sources,
        targets,
        target_deltas,
        rng_key,
    ):
        return update_model(
            model,
            params,
            opt_state,
            optimizer,
            sources,
            targets,
            target_deltas,
            rng_key,
        )

    step = 1
    if FLAGS.checkpoint_path is not None:
        params, opt_state, step = load_checkpoint(checkpointer, FLAGS.checkpoint_path)

    while step < FLAGS.steps + 1:
        for clips in tqdm(batch_iterator(train_dataset, FLAGS.batch_size)):
            metrics = {}

            offset_key, rng_key = jax.random.split(rng_key)
            src, tgt, offsets = prepare_rvm_src_tgt_pairs(
                offset_key, clips, FLAGS.src_frames, FLAGS.tgt_frames
            )

            train_key, rng_key = jax.random.split(rng_key)
            params, opt_state, train_metrics = training_step(
                params,
                opt_state,
                src,
                tgt,
                offsets,
                train_key,
            )
            metrics.update(train_metrics)

            if step % FLAGS.eval_interval == 0:
                eval_key, rng_key = jax.random.split(rng_key)
                eval_metrics = full_evaluation(
                    model,
                    test_dataset,
                    params,
                    FLAGS.src_frames,
                    FLAGS.tgt_frames,
                    FLAGS.batch_size,
                    eval_key
                )
                metrics.update(eval_metrics)

            if step % FLAGS.save_interval == 0:
                state = {
                    'step': step,
                    'params': params,
                    'opt_state': opt_state,
                }
                checkpointer.save(
                    os.path.abspath(os.path.join(FLAGS.save_dir, f'checkpoint_{step}')),
                    state,
                )

            wandb.log(metrics, step=step)

            step += 1
            if step > FLAGS.steps:
                break


if __name__ == '__main__':
    app.run(main)
