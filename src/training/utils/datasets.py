import zarr
import numpy as np
from torch.utils.data import Dataset


def normalize_video(video):
    video = video.astype("float32")

    for c in range(video.shape[1]):  # per channel
        p1, p99 = np.percentile(video[:, c], (1, 99))
        video[:, c] = np.clip(video[:, c], p1, p99)
        video[:, c] = (video[:, c] - p1) / (p99 - p1)

    return video


class ZarrVideoDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        dataset_key="Data",
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.dataset_key = dataset_key

    def __len__(self):
        return self.root[self.dataset_key].shape[0]

    def __getitem__(self, idx):
        video = self.root[self.dataset_key][idx, ...]  # (T,C,H,W)

        video = normalize_video(video)

        return video
