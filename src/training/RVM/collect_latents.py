import numpy as np
from absl import app, flags
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import random
import os
import json
import pickle

from utils.model import get_rvm
from utils.dataloader import create_train_val_datasets, TransformPipeline, batch_iterator


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('config_path', None,
                    'Path to the configuration file.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to the model checkpoint.')
flags.DEFINE_string('output_dir', 'latents', 'Directory to save the collected latents.')
flags.DEFINE_integer('clip_frames', 64, 'Number of frames in each video clip.')
flags.DEFINE_integer('batch_size', 16, 'Batch size for processing video clips.')
flags.DEFINE_integer('num_samples', 5000, 'Number of samples to extract latents from.')


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)


def load_checkpoint(checkpointer, checkpoint_path):
    state = checkpointer.restore(os.path.abspath(checkpoint_path))
    return state['params'], state['opt_state'], state['step']


def extract_latents(
    model,
    test_dataset,
    params,
    batch_size,
    num_samples
):
    @jax.jit
    def forward(sources):
        output = model.apply(
            {"params": params},
            sources,
            method=model.encode,
        )

        output = jnp.mean(output, axis=-2)  # Spatial averaging

        return output

    features = []
    metadata = []
    num_samples = 0
    loader = batch_iterator(test_dataset, batch_size=batch_size, aux=True)
    while num_samples < FLAGS.num_samples:
        for batch in loader:
            clips = batch["clips"]
            meta = batch["metadata"]
            clips = np.transpose(clips, (0, 1, 3, 4, 2))

            features = forward(clips)

            features.extend(np.array(features))
            metadata.extend(meta)

            num_samples += len(clips)

            if num_samples >= FLAGS.num_samples:
                break

            print(f"Extracted {num_samples} samples...")

    features = np.array(features)

    return features, metadata


def main(_):
    with open(FLAGS.config_path, 'r') as f:
        config = json.load(f)

    set_seed(FLAGS.seed)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    model = get_rvm(
        num_channels=len(config['channel_names'].split()),
        masking_ratio=config['masking_ratio'],
        variant=config['rvm_variant'],
    )

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    model_params, _, _ = load_checkpoint(checkpointer, FLAGS.checkpoint_path)

    transform_pipeline = TransformPipeline(
        transform_names_list=config['transforms'].split(),
        arcsinh_cofactor=config['arcsinh_cofactor'],
        butterworth_cutoff=config['butterworth_cutoff'],
        butterworth_order=config['butterworth_order'],
        per_frame_butterworth=config['per_frame_butterworth'],
    )

    _, val_dataset = create_train_val_datasets(
        config['dataset_path'],
        flags.clip_frames,
        config['clip_size'],
        config['acq_freq'],
        config['channel_names'].split(),
        transform_pipeline,
        random_crop_val=True
    )

    features, metadata = extract_latents(
        model,
        val_dataset,
        model_params,
        FLAGS.batch_size,
        FLAGS.num_samples,
    )

    features_path = os.path.join(FLAGS.output_dir, 'features.npy')
    features_metadata_path = os.path.join(FLAGS.output_dir, 'metadata.pkl')

    np.save(features_path, features)
    with open(features_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    app.run(main)
