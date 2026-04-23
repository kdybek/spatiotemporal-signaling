import zarr
import numpy as np
from torch.utils.data import Dataset
import math


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


def get_clip(root, video_name, clip_frames, clip_size, acq_freq, channel_names_list, validation=False):
    video_shape = root[video_name].shape
    video_metadata = root[video_name].attrs["Metadata"]
    channel_indices = [video_metadata[name] for name in channel_names_list]
    T, C, H, W = video_shape
    original_acq_freq = video_metadata["Acq_freq_min"]
    magnification = video_metadata["Magnification"]
    step = acq_freq / original_acq_freq

    if not math.isclose(step, round(step), tolerance=1e-5):
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

    start_t = np.random.randint(0, T - min_frames_needed + 1)
    start_h = np.random.randint(0, H - clip_size + 1)
    start_w = np.random.randint(0, W - clip_size + 1)

    clip = root[video_name][start_t:start_t + min_frames_needed:step, channel_indices,
                            start_h:start_h + clip_size, start_w:start_w + clip_size]

    if downsample_2x:
        clip = downsample_video_2x(clip)

    return clip


class ZarrVideoDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        clip_frames,
        clip_size,
        acq_freq,
        channel_names,
        transform_names,
        augment,
        arcsinh_cofactor=None,
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.acq_freq = acq_freq
        self.channel_names_list = channel_names.split()
        self.transform_names_list = transform_names.split()
        self.arcsinh_cofactor = arcsinh_cofactor
        self.augment = augment

        self._legal_video_names = []

        for video_name in self.root:
            try:
                get_clip(
                    self.root,
                    video_name,
                    self.clip_frames,
                    self.clip_size,
                    self.acq_freq,
                    self.channel_names_list,
                    validation=True,
                )
            except ValueError as e:
                print(f"Skipping video {video_name} due to error: {e}")
                continue

            self._legal_video_names.append(video_name)

    def __len__(self):
        return len(self._legal_video_names)

    def __getitem__(self, idx):
        video_name = self._legal_video_names[idx]
        video = get_clip(
            self.root,
            video_name,
            self.clip_frames,
            self.clip_size,
            self.acq_freq,
            self.channel_names_list,
        )

        video = video.astype("float32")

        for transform_name in self.transform_names_list:
            if transform_name == "percentile_norm":
                video = percentile_norm(video)
            if transform_name == "arcsinh":
                if self.arcsinh_cofactor is None:
                    raise ValueError(
                        "arcsinh_cofactor must be provided for arcsinh transform.")
                video = np.arcsinh(video / self.arcsinh_cofactor)
            if transform_name == "log1p":
                video = np.log1p(video)
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                video = video[:, :, :, ::-1]

            # Random vertical flip
            if np.random.rand() < 0.5:
                video = video[:, :, ::-1, :]

        return video
