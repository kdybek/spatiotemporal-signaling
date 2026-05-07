import zarr
import numpy as np
from torch.utils.data import Dataset
import math
from pathlib import Path

def percentile_norm(video):
    for c in range(video.shape[1]):  # per channel
        p1, p99 = np.percentile(video[:, c], (1, 99))
        video[:, c] = np.clip(video[:, c], p1, p99)
        video[:, c] = (video[:, c] - p1) / (p99 - p1)

    return video


def downsample_video_2x(video):
    # Average 2x2 blocks to downsample by a factor of 2 in height and width
    return (
        video[:, :, 0::2, 0::2] +
        video[:, :, 0::2, 1::2] +
        video[:, :, 1::2, 0::2] +
        video[:, :, 1::2, 1::2]
    ) / 4.0


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

    clip = root[video_name][start_t:start_t + min_frames_needed:step, channel_indices,
                            start_h:start_h + clip_size, start_w:start_w + clip_size]

    if downsample_2x:
        clip = downsample_video_2x(clip)

    return clip


def create_train_test_datasets(
    test_fraction,
    zarr_path,
    clip_frames_train,
    clip_frames_test,
    clip_size,
    acq_freq,
    channel_names,
    transform_names,
    arcsinh_cofactor=None
):
    assert 0 < test_fraction < 1, "test_fraction must be between 0 and 1"

    root = zarr.open(zarr_path, mode="r")
    channel_names_list = channel_names.split()
    train_video_names = []
    test_video_names = []
    for video_name in root:
        try:
            get_clip(
                root,
                video_name,
                clip_frames_train,
                clip_size,
                acq_freq,
                channel_names_list,
                random_crop=False,
                validation=True,
            )
        except ValueError as e:
            print(f"Skipping video {video_name} for train set due to error: {e}")
            continue

        train_video_names.append(video_name)

    for video_name in root:
        try:
            get_clip(
                root,
                video_name,
                clip_frames_test,
                clip_size,
                acq_freq,
                channel_names_list,
                random_crop=False,
                validation=True,
            )
        except ValueError as e:
            print(f"Skipping video {video_name} for test set due to error: {e}")
            continue

        test_video_names.append(video_name)

    print(f"Found {len(train_video_names)} viable videos for train set and {len(test_video_names)} for test set out of {len(root)} total videos.")

    test_video_names = np.random.permutation(test_video_names)
    test_split = int(test_fraction * len(test_video_names))
    test_video_names = test_video_names[:test_split]

    train_video_names = [name for name in train_video_names if name not in test_video_names]

    print(f"Using {len(train_video_names)} videos for training and {len(test_video_names)} videos for testing.")

    train_dataset = ZarrVideoDataset(
        root,
        train_video_names,
        clip_frames_train,
        clip_size,
        acq_freq,
        channel_names,
        transform_names,
        augment=True,
        random_crop=True,
        arcsinh_cofactor=arcsinh_cofactor,
    )

    test_dataset = ZarrVideoDataset(
        root,
        test_video_names,
        clip_frames_test,
        clip_size,
        acq_freq,
        channel_names,
        transform_names,
        augment=False,
        random_crop=False,
        arcsinh_cofactor=arcsinh_cofactor,
    )

    return train_dataset, test_dataset


class ZarrVideoDataset(Dataset):
    def __init__(
        self,
        root,
        video_names,
        clip_frames,
        clip_size,
        acq_freq,
        channel_names,
        transform_names,
        augment,
        random_crop,
        arcsinh_cofactor=None,
    ):
        self.root = root
        self.video_names = video_names
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.acq_freq = acq_freq
        self.channel_names_list = channel_names.split()
        self.transform_names_list = transform_names.split()
        self.arcsinh_cofactor = arcsinh_cofactor
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

        for transform_name in self.transform_names_list:
            if transform_name == "percentile_norm":
                video = percentile_norm(video)
            elif transform_name == "arcsinh":
                if self.arcsinh_cofactor is None:
                    raise ValueError(
                        "arcsinh_cofactor must be provided for arcsinh transform.")
                video = np.arcsinh(video / self.arcsinh_cofactor)
            elif transform_name == "log1p":
                video = np.log1p(video)
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

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
