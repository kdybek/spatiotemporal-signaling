import numpy as np
from tqdm import tqdm
from absl import app, flags
import random
import os
import wandb
from collections import defaultdict

from utils.logging import get_exp_name, setup_wandb
from utils.dataloader import (
    create_train_val_datasets,
    TransformPipeline,
    batch_iterator,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")

flags.DEFINE_integer("sampling_epochs", 10_000, "Number of sampling epochs.")

flags.DEFINE_string("dataset_path", "toy_dataset.zarr", "Path to the train dataset.")

flags.DEFINE_integer("clip_size", 256, "Height and width of input images.")
flags.DEFINE_integer("clip_frames", 1, "Number of frames in each video clip.")
flags.DEFINE_float(
    "acq_freq", 15.0, "Acquisition frequency (in minutes) for sampling video clips."
)
flags.DEFINE_string(
    "channel_names",
    "Ch_H2B Ch_ERK-KTR Ch_AKT-KTR Ch_Geminin",
    "Space-separated list of channel names to use from the videos.",
)

flags.DEFINE_string(
    "transforms",
    "arcsinh butterworth",
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


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)


def visualize(
    clips, channel_inds, channel_names, experiment_names
):
    metrics = defaultdict(list)

    C = clips.shape[-1]
    for c in range(C):
        for i in range(clips.shape[0]):
            channel_ind = channel_inds[i, c]
            experiment_name = experiment_names[i]
            image = clips[i, 0, ..., c]
            image -= image.min()
            image /= max(image.max(), 1e-8)
            image = (image * 255).astype(np.uint8)
            metrics[
                f"visualizations"
            ].append(wandb.Image(image, caption=f"{experiment_name} - {channel_names[channel_ind]}"))

    return metrics


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        x = np.asarray(x).ravel().copy()

        n = x.size
        if n == 0:
            return

        mean = x.mean()
        M2 = ((x - mean) ** 2).sum()

        if self.n == 0:
            self.n = n
            self.mean = mean
            self.M2 = M2
            return

        delta = mean - self.mean
        new_n = self.n + n

        self.mean += delta * n / new_n
        self.M2 += M2 + delta**2 * self.n * n / new_n
        self.n = new_n

    @property
    def std(self):
        if self.n < 2:
            return float("nan")
        return np.sqrt(self.M2 / (self.n - 1))

    def to_dict(self):
        return {"mean": self.mean, "std": self.std, "n": self.n}


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        project="Spaciotemporal Signaling", group=FLAGS.run_group, name=exp_name
    )
    set_seed(FLAGS.seed)

    transform_pipeline = TransformPipeline(
        transform_names_list=FLAGS.transforms.split(),
        arcsinh_cofactor=FLAGS.arcsinh_cofactor,
        butterworth_cutoff=FLAGS.butterworth_cutoff,
        butterworth_order=FLAGS.butterworth_order,
        per_frame_butterworth=FLAGS.per_frame_butterworth,
    )

    dummy_downstream_split_config = {
        "train": None,
        "test": None,
    }

    train_dataset, _, _ = (
        create_train_val_datasets(
            zarr_path=FLAGS.dataset_path,
            clip_frames=FLAGS.clip_frames,
            clip_size=FLAGS.clip_size,
            acq_freq=FLAGS.acq_freq,
            channel_names_list=FLAGS.channel_names.split(),
            transform_pipeline=transform_pipeline,
            downstream_split_config=dummy_downstream_split_config,
        )
    )

    channel_names = FLAGS.channel_names.split()

    channel_stats = {
        name: RunningStats()
        for name in channel_names
    }

    experiment_channel_stats = defaultdict(
        lambda: {
            name: RunningStats()
            for name in channel_names
        }
    )

    for epoch in range(FLAGS.sampling_epochs):
        loader = batch_iterator(train_dataset, batch_size=1, shuffle=False)
        all_clips = []
        all_channel_inds = []
        all_experiment_names = []
        for clips, channel_inds, metadata in tqdm(loader, desc="Sampling epoch"):
            clips = clips.squeeze(0)
            channel_inds = channel_inds.squeeze(0)
            clips = clips.transpose(0, 2, 3, 1)  # (T, H, W, C)
            clips = clips[..., channel_inds]
            metadata = metadata[0]
            experiment_name = os.path.basename(metadata["Path"])

            for c, channel_idx in enumerate(channel_inds):
                channel_name = channel_names[channel_idx]
                values = clips[..., c]

                channel_stats[channel_name].update(values)
                experiment_channel_stats[experiment_name][channel_name].update(values)

            if epoch == 0:
                all_clips.append(clips)
                all_channel_inds.append(channel_inds)
                all_experiment_names.append(experiment_name)

        if epoch % 100 == 0:
            metrics = {}

            for channel_name, stats in channel_stats.items():
                for k, v in stats.to_dict().items():
                    metrics[f"global/{channel_name}/{k}"] = v

            for experiment, stats_dict in experiment_channel_stats.items():
                for channel_name, stats in stats_dict.items():
                    for k, v in stats.to_dict().items():
                        metrics[
                            f"{experiment}/{channel_name}/{k}"
                        ] = v

            if epoch == 0:
                all_clips = np.stack(all_clips, axis=0)
                all_channel_inds = np.stack(all_channel_inds, axis=0)

                images = visualize(
                    clips=all_clips,
                    channel_inds=all_channel_inds,
                    channel_names=channel_names,
                    experiment_names=all_experiment_names,
                )

                metrics.update(images)

            wandb.log(metrics)


if __name__ == "__main__":
    app.run(main)
