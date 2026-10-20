import yaml
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

from utils.model import get_rvm, get_optimizer, BaselinePatcher
from utils.logging import get_exp_name, setup_wandb
from utils.dataloader import (
    create_train_val_datasets,
    TransformPipeline,
    prepare_rvm_src_tgt_pairs,
    batch_iterator,
)
from utils.evaluation import partial_evaluation, downstream_evaluation
from utils.loss import update_model

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")

flags.DEFINE_integer("steps", 250_000, "Number of training steps.")
flags.DEFINE_integer("eval_steps", 100, "Number of steps for evaluation.")
flags.DEFINE_integer("eval_interval", 25_000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 25_000, "Saving interval.")

flags.DEFINE_integer("downstream_interval", 50_000, "Downstream training interval.")
flags.DEFINE_integer("downstream_train_steps", 10_000, "Number of steps for downstream training.")
flags.DEFINE_integer("downstream_eval_interval", 2_000, "Downstream evaluation interval.")
flags.DEFINE_integer("downstream_eval_steps", 100, "Number of steps for downstream evaluation.")

flags.DEFINE_string("dataset_path", "toy_dataset.zarr", "Path to the train dataset.")
flags.DEFINE_string(
    "downsteam_dataset_split_config",
    "downstream_split.yaml",
    "Path to the downstream dataset split config file.",
)
flags.DEFINE_string("save_dir", "checkpoints", "Directory to save model checkpoints.")

flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("batch_size", 4, "Batch size for training and evaluation.")
flags.DEFINE_integer(
    "grad_accumulation_steps",
    4,
    "Number of steps to accumulate gradients before updating the model.",
)
flags.DEFINE_integer("clip_size", 256, "Height and width of input images.")
flags.DEFINE_integer("clip_frames", 64, "Number of frames in each video clip.")
flags.DEFINE_float(
    "acq_freq", 15.0, "Acquisition frequency (in minutes) for sampling video clips."
)
flags.DEFINE_string(
    "channel_names",
    "Ch_H2B Ch_ERK-KTR Ch_AKT-KTR Ch_Geminin",
    "Space-separated list of channel names to use from the videos.",
)

flags.DEFINE_integer("src_frames", 4, "Number of source frames for reconstruction.")
flags.DEFINE_integer("tgt_frames", 4, "Number of target frames for reconstruction.")
flags.DEFINE_integer(
    "src_sample_prefix", 16, "Clip prefix length for source frames sampling."
)
flags.DEFINE_integer(
    "min_offset", 4, "Minimum offset between source and target frames."
)
flags.DEFINE_integer(
    "max_offset", 48, "Maximum offset between source and target frames."
)
flags.DEFINE_float(
    "src_masking_ratio", 0.0, "Ratio of source tokens to mask during training."
)
flags.DEFINE_float(
    "tgt_masking_ratio", 0.95, "Ratio of target tokens to mask during training."
)
flags.DEFINE_string("rvm_variant", "S", "Variant of the RVM to use.")

flags.DEFINE_string(
    "transforms",
    "arcsinh butterworth percentile_norm",
    "Space-separated list of transforms to apply to the videos. Supported: percentile_norm, arcsinh, log1p, butterworth.",
)
flags.DEFINE_float("arcsinh_cofactor", 5.0, "Cofactor for arcsinh transform.")
flags.DEFINE_float(
    "butterworth_cutoff", 0.2, "Cutoff frequency for Butterworth low-pass filter."
)
flags.DEFINE_integer("butterworth_order", 2, "Order of Butterworth low-pass filter.")
flags.DEFINE_boolean(
    "per_frame_butterworth",
    True,
    "Whether to apply Butterworth filter independently to each frame (instead of across time).",
)

flags.DEFINE_string(
    "checkpoint_path", None, "Path to a checkpoint to load model parameters from."
)

flags.DEFINE_boolean("baseline_model", False, "Whether to use a naive baseline model.")


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)


def load_split_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize_model(model, rng_key):
    num_channels = len(FLAGS.channel_names.split())
    dummy_src = jnp.ones(
        (1, FLAGS.src_frames, FLAGS.clip_size, FLAGS.clip_size, num_channels),
        dtype=jnp.float32,
    )
    dummy_tgt = jnp.ones(
        (1, FLAGS.tgt_frames, FLAGS.clip_size, FLAGS.clip_size, num_channels),
        dtype=jnp.float32,
    )
    dummy_channel_inds = jnp.arange(num_channels, dtype=jnp.int32)[None, :]
    dummy_deltas = jnp.zeros((1, FLAGS.tgt_frames), dtype=jnp.int32)

    params = model.init(
        rng_key, dummy_src, dummy_tgt, dummy_channel_inds, dummy_deltas
    )["params"]
    return params


def load_checkpoint(checkpointer, checkpoint_path):
    state = checkpointer.restore(os.path.abspath(checkpoint_path))
    return state["params"], state["opt_state"], state["step"]


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        project="Spaciotemporal Signaling", group=FLAGS.run_group, name=exp_name
    )
    set_seed(FLAGS.seed)
    rng_key = jax.random.PRNGKey(FLAGS.seed)

    transform_pipeline = TransformPipeline(
        transform_names_list=FLAGS.transforms.split(),
        arcsinh_cofactor=FLAGS.arcsinh_cofactor,
        butterworth_cutoff=FLAGS.butterworth_cutoff,
        butterworth_order=FLAGS.butterworth_order,
        per_frame_butterworth=FLAGS.per_frame_butterworth,
    )

    downstream_split_config = load_split_config(FLAGS.downsteam_dataset_split_config)

    train_dataset, val_dataset, downstream_train_dataset, downstream_test_dataset = (
        create_train_val_datasets(
            zarr_path=FLAGS.dataset_path,
            clip_frames=FLAGS.clip_frames,
            clip_size=FLAGS.clip_size,
            acq_freq=FLAGS.acq_freq,
            channel_names_list=FLAGS.channel_names.split(),
            transform_pipeline=transform_pipeline,
            downstream_split_config=downstream_split_config,
        )
    )

    if FLAGS.baseline_model:
        baseline_model = BaselinePatcher(patch_size=(16, 16))
        eval_key, rng_key = jax.random.split(rng_key)
        downstream_metrics = downstream_evaluation(
            foundation_model=baseline_model,
            foundation_model_params={},
            downstream_train_dataset=downstream_train_dataset,
            downstream_test_dataset=downstream_test_dataset,
            downstream_train_steps=FLAGS.downstream_train_steps,
            downstream_eval_interval=FLAGS.downstream_eval_interval,
            downstream_eval_steps=FLAGS.downstream_eval_steps,
            batch_size=FLAGS.batch_size,
            base_lr=FLAGS.learning_rate,
            grad_accumulation_steps=FLAGS.grad_accumulation_steps,
            main_training_step=0,
            rng_key=eval_key,
        )
        return  # Exit after evaluating the baseline model

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flags_dict = FLAGS.flag_values_dict()

    with open(os.path.join(FLAGS.save_dir, "config.json"), "w") as f:
        json.dump(flags_dict, f, indent=4)

    model = get_rvm(
        num_channels=len(FLAGS.channel_names.split()),
        src_masking_ratio=FLAGS.src_masking_ratio,
        tgt_masking_ratio=FLAGS.tgt_masking_ratio,
        variant=FLAGS.rvm_variant,
    )

    init_key, rng_key = jax.random.split(rng_key)
    params = initialize_model(model, init_key)

    optimizer = get_optimizer(
        base_lr=FLAGS.learning_rate,
        warmup_steps=int(0.05 * FLAGS.steps),
        total_steps=FLAGS.steps,
        grad_accumulation_steps=FLAGS.grad_accumulation_steps,
    )
    opt_state = optimizer.init(params)

    wandb.define_metric("fd_step")
    wandb.define_metric("*", step_metric="fd_step")

    if FLAGS.checkpoint_path is None:
        eval_key, rng_key = jax.random.split(rng_key)
        init_metrics = partial_evaluation(
            model,
            val_dataset,
            params,
            FLAGS.src_frames,
            FLAGS.tgt_frames,
            FLAGS.channel_names.split(),
            FLAGS.src_sample_prefix,
            FLAGS.min_offset,
            FLAGS.max_offset,
            FLAGS.batch_size,
            FLAGS.eval_steps,
            eval_key,
        )
        downstream_eval_key, rng_key = jax.random.split(rng_key)
        downstream_metrics = downstream_evaluation(
            foundation_model=model,
            foundation_model_params=params,
            downstream_train_dataset=downstream_train_dataset,
            downstream_test_dataset=downstream_test_dataset,
            downstream_train_steps=FLAGS.downstream_train_steps,
            downstream_eval_interval=FLAGS.downstream_eval_interval,
            downstream_eval_steps=FLAGS.downstream_eval_steps,
            batch_size=FLAGS.batch_size,
            base_lr=FLAGS.learning_rate,
            grad_accumulation_steps=FLAGS.grad_accumulation_steps,
            main_training_step=0,
            rng_key=downstream_eval_key,
        )
        init_metrics.update(downstream_metrics)
        init_metrics.update({"fd_step": 0})
        wandb.log(init_metrics)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    @jax.jit
    def training_step(
        params,
        opt_state,
        sources,
        targets,
        channel_inds,
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
            channel_inds,
            target_deltas,
            rng_key,
        )

    step = 1
    if FLAGS.checkpoint_path is not None:
        params, opt_state, step = load_checkpoint(checkpointer, FLAGS.checkpoint_path)

    while step <= FLAGS.steps:
        loader = batch_iterator(train_dataset, batch_size=FLAGS.batch_size)
        for clips, channel_inds, _ in tqdm(loader, desc="Training epoch"):
            metrics = {}

            src, tgt, offsets = prepare_rvm_src_tgt_pairs(
                clips,
                FLAGS.src_frames,
                FLAGS.tgt_frames,
                FLAGS.src_sample_prefix,
                FLAGS.min_offset,
                FLAGS.max_offset,
            )

            train_key, rng_key = jax.random.split(rng_key)
            params, opt_state, train_metrics = training_step(
                params,
                opt_state,
                src,
                tgt,
                channel_inds,
                offsets,
                train_key,
            )
            metrics.update(train_metrics)

            if step % FLAGS.eval_interval == 0:
                eval_key, rng_key = jax.random.split(rng_key)
                eval_metrics = partial_evaluation(
                    model,
                    val_dataset,
                    params,
                    FLAGS.src_frames,
                    FLAGS.tgt_frames,
                    FLAGS.channel_names.split(),
                    FLAGS.src_sample_prefix,
                    FLAGS.min_offset,
                    FLAGS.max_offset,
                    FLAGS.batch_size,
                    FLAGS.eval_steps,
                    eval_key,
                )
                metrics.update(eval_metrics)

            if step % FLAGS.downstream_interval == 0:
                downstream_eval_key, rng_key = jax.random.split(rng_key)
                downstream_metrics = downstream_evaluation(
                    foundation_model=model,
                    foundation_model_params=params,
                    downstream_train_dataset=downstream_train_dataset,
                    downstream_test_dataset=downstream_test_dataset,
                    downstream_train_steps=FLAGS.downstream_train_steps,
                    downstream_eval_interval=FLAGS.downstream_eval_interval,
                    downstream_eval_steps=FLAGS.downstream_eval_steps,
                    batch_size=FLAGS.batch_size,
                    base_lr=FLAGS.learning_rate,
                    grad_accumulation_steps=FLAGS.grad_accumulation_steps,
                    main_training_step=step,
                    rng_key=downstream_eval_key,
                )
                metrics.update(downstream_metrics)

            if step % FLAGS.save_interval == 0:
                state = {
                    "step": step,
                    "params": params,
                    "opt_state": opt_state,
                }
                checkpointer.save(
                    os.path.abspath(os.path.join(FLAGS.save_dir, f"checkpoint_{step}")),
                    state,
                )

            metrics.update({"fd_step": step})
            wandb.log(metrics)

            step += 1
            if step > FLAGS.steps:
                break


if __name__ == "__main__":
    app.run(main)
