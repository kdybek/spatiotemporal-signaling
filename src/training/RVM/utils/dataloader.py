import zarr
import numpy as np
import jax.numpy as jnp
import random
from skimage.filters import butterworth
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread


DATASET_T_CHUNK = 64
DATASET_H_CHUNK = 256
DATASET_W_CHUNK = 256


def percentile_norm(video):
    for c in range(video.shape[1]):  # per channel
        p1, p99 = np.percentile(video[:, c], (1, 99))
        video[:, c] = np.clip(video[:, c], p1, p99)
        video[:, c] = (video[:, c] - p1) / (p99 - p1)

    return video


def butterworth_filter(video, cutoff, order, per_frame):
    if per_frame:
        filtered_video = np.empty_like(video)
        for t in range(video.shape[0]):
            for c in range(video.shape[1]):
                filtered_video[t, c] = butterworth(
                    video[t, c],
                    cutoff_frequency_ratio=cutoff,
                    order=order,
                    high_pass=False
                )
        return filtered_video
    else:
        # Apply the filter across the entire video (treating it as a 3D volume)
        filtered_video = np.empty_like(video)
        for c in range(video.shape[1]):
            filtered_video[:, c] = butterworth(
                video[:, c],
                cutoff_frequency_ratio=cutoff,
                order=order,
                high_pass=False
            )
        return filtered_video


def downsample_video_2x(video):
    # Average 2x2 blocks to downsample by a factor of 2 in height and width
    return (
        video[:, :, 0::2, 0::2] +
        video[:, :, 0::2, 1::2] +
        video[:, :, 1::2, 0::2] +
        video[:, :, 1::2, 1::2]
    ) / 4.0


def snap_to_chunk_size(value, chunk_size):
    return (value // chunk_size) * chunk_size


def get_clip(root, video_name, clip_frames, clip_size, acq_freq, channel_names_list, random_crop, validation=False):
    video_shape = root[video_name].shape
    video_metadata = root[video_name].attrs["Metadata"]
    channel_indices = [video_metadata[name] for name in channel_names_list]
    T, C, H, W = video_shape
    original_acq_freq = video_metadata["Acq_freq_min"]
    magnification = video_metadata["Magnification"]
    step = acq_freq / original_acq_freq

    if not math.isclose(step, round(step), abs_tol=1e-5):
        raise ValueError(f"Acq_freq {acq_freq} is not an integer multiple of video Acq_freq_min {
                         original_acq_freq}")

    step = int(round(step))

    if magnification == 40:
        downsample_2x = True
        clip_size *= 2  # We need to double the clip size to get the same field of view
    elif magnification == 20:
        downsample_2x = False
    else:
        raise ValueError(f"Unsupported Magnification {magnification}")

    # Total frames needed to get clip_frames with the given step
    min_frames_needed = clip_frames * step - (step - 1)
    if T < min_frames_needed or H < clip_size or W < clip_size:
        raise ValueError(f"Video shape {video_shape} is smaller than clip requirements: "
                         f"clip_frames={clip_frames}, clip_size={clip_size}")

    if validation:  # This is ugly, but all the things that can be wrong with the video should be caught up to this point
        return

    if random_crop:
        start_t = np.random.randint(0, T - min_frames_needed + 1)
        start_h = np.random.randint(0, H - clip_size + 1)
        start_w = np.random.randint(0, W - clip_size + 1)
    else:
        start_t = (T - min_frames_needed) // 2
        start_h = (H - clip_size) // 2
        start_w = (W - clip_size) // 2

    start_t = snap_to_chunk_size(start_t, DATASET_T_CHUNK)
    start_h = snap_to_chunk_size(start_h, DATASET_H_CHUNK)
    start_w = snap_to_chunk_size(start_w, DATASET_W_CHUNK)

    clip = root[video_name][start_t:start_t + min_frames_needed:step, channel_indices,
                            start_h:start_h + clip_size, start_w:start_w + clip_size]

    if downsample_2x:
        clip = downsample_video_2x(clip)

    return clip


class TransformPipeline:
    def __init__(self, transform_names_list, arcsinh_cofactor, butterworth_cutoff, butterworth_order, per_frame_butterworth):
        self.transform_names_list = transform_names_list
        self.arcsinh_cofactor = arcsinh_cofactor
        self.butterworth_cutoff = butterworth_cutoff
        self.butterworth_order = butterworth_order
        self.per_frame_butterworth = per_frame_butterworth

    def __call__(self, video):
        for transform_name in self.transform_names_list:
            if transform_name == 'percentile_norm':
                video = percentile_norm(video)
            elif transform_name == 'arcsinh':
                video = np.arcsinh(video / self.arcsinh_cofactor)
            elif transform_name == 'log1p':
                video = np.log1p(video)
            elif transform_name == 'butterworth':
                video = butterworth_filter(
                    video, cutoff=self.butterworth_cutoff, order=self.butterworth_order, per_frame=self.per_frame_butterworth)
            else:
                raise ValueError(f"Unsupported transform: {transform_name}")

        return video


class ZarrVideoDataset():
    def __init__(
        self,
        root,
        video_names,
        clip_frames,
        clip_size,
        acq_freq,
        channel_names_list,
        transform_pipeline,
        augment,
        random_crop,
    ):
        self.root = root
        self.video_names = video_names
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.acq_freq = acq_freq
        self.channel_names_list = channel_names_list
        self.transform_pipeline = transform_pipeline
        self.augment = augment
        self.random_crop = random_crop

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video = get_clip(
            self.root,
            video_name,
            self.clip_frames,
            self.clip_size,
            self.acq_freq,
            self.channel_names_list,
            self.random_crop,
        )

        video = video.astype("float32")
        video = self.transform_pipeline(video)

        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                video = video[:, :, :, ::-1].copy()

            # Random vertical flip
            if np.random.rand() < 0.5:
                video = video[:, :, ::-1, :].copy()

        path = self.root[video_name].attrs["Metadata"]["Path"]
        exp_name = Path(path).name

        return video, exp_name


def create_train_test_datasets(
    test_fraction,
    zarr_path,
    clip_frames,
    clip_size,
    acq_freq,
    channel_names_list,
    transform_pipeline,
):
    assert 0 < test_fraction < 1, "test_fraction must be between 0 and 1"

    root = zarr.open(zarr_path, mode="r")
    video_names = []
    for video_name in root:
        try:
            get_clip(
                root,
                video_name,
                clip_frames,
                clip_size,
                acq_freq,
                channel_names_list,
                random_crop=False,
                validation=True,
            )
        except Exception as e:
            print(f"Skipping video {video_name} due to error: {e}")
            continue

        video_names.append(video_name)

    print(f"Found {len(video_names)} viable videos out of {len(root)} total videos.")

    random.shuffle(video_names)
    test_split = int(test_fraction * len(video_names))
    test_video_names = video_names[:test_split]
    train_video_names = video_names[test_split:]

    print(f"Using {len(train_video_names)} videos for training and {
          len(test_video_names)} videos for testing.")

    train_dataset = ZarrVideoDataset(
        root,
        train_video_names,
        clip_frames,
        clip_size,
        acq_freq,
        channel_names_list,
        transform_pipeline,
        augment=True,
        random_crop=True,
    )

    test_dataset = ZarrVideoDataset(
        root,
        test_video_names,
        clip_frames,
        clip_size,
        acq_freq,
        channel_names_list,
        transform_pipeline,
        augment=False,
        random_crop=False,
    )

    return train_dataset, test_dataset


def batch_iterator(dataset, batch_size, shuffle=True, exp_name=False, max_workers=64):
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]

            samples = list(executor.map(dataset.__getitem__, batch_idx))

            clips, exp_names = zip(*samples)

            clips = np.stack(clips)
            exp_names = list(exp_names)

            if exp_name:
                yield clips, exp_names
            else:
                yield clips


def prefetch(generator, buffer_size):
    queue = Queue(maxsize=buffer_size)

    def worker():
        for item in generator:
            queue.put(item)
        queue.put(None)

    Thread(target=worker, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item


def prepare_rvm_src_tgt_pairs(
        clip_batch,
        src_frames,
        tgt_frames,
        src_sample_prefix,
        min_offset,
        max_offset
):
    """
    Prepares source and target pairs for RVM training.

    Args:
        clip_batch: A batch of video clips with shape (B, T, C, H, W).
        src_frames: Number of source frames to select.
        tgt_frames: Number of target frames to select.
        src_sample_prefix: Prefix length for source sample selection.
        min_offset: Minimum offset for target frame selection.
        max_offset: Maximum offset for target frame selection.

    Returns:
        src_batch: A batch of source frames.
        tgt_batch: A batch of target frames.
        offsets: A batch of offsets used for target frames.
    """
    # (B, T, C, H, W) -> (B, T, H, W, C)
    clip_batch = np.transpose(clip_batch, (0, 1, 3, 4, 2))

    B, T = clip_batch.shape[:2]

    assert src_sample_prefix + max_offset <= T
    assert src_frames <= src_sample_prefix

    max_src_init_idx = src_sample_prefix - src_frames
    init_src_idx = np.random.randint(
        low=0,
        high=max_src_init_idx + 1,
        size=(B,),
    )

    src_idx = init_src_idx[:, None] + jnp.arange(src_frames)[None, :]
    src_batch = clip_batch[np.arange(B)[:, None], src_idx]

    offsets = np.random.randint(
        low=min_offset,
        high=max_offset + 1,
        size=(B, tgt_frames),
    )

    tgt_idx = src_idx[:, -1:] + offsets
    tgt_batch = clip_batch[np.arange(B)[:, None], tgt_idx]

    src_batch = jnp.asarray(src_batch)
    tgt_batch = jnp.asarray(tgt_batch)
    offsets = jnp.asarray(offsets)

    return src_batch, tgt_batch, offsets
