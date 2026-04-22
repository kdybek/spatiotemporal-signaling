import zarr
import numpy as np
from torch.utils.data import Dataset


def percentile_norm(video):
    for c in range(video.shape[1]):  # per channel
        p1, p99 = np.percentile(video[:, c], (1, 99))
        video[:, c] = np.clip(video[:, c], p1, p99)
        video[:, c] = (video[:, c] - p1) / (p99 - p1)

    return video


class ZarrVideoDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        dataset_key,
        transform_names,
        augment,
        arcsinh_cofactor=None,
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.dataset_key = dataset_key
        self.transform_names = transform_names
        self.arcsinh_cofactor = arcsinh_cofactor
        self.augment = augment

    def __len__(self):
        return self.root[self.dataset_key].shape[0]

    def __getitem__(self, idx):
        video = self.root[self.dataset_key][idx, ...]  # (T,C,H,W)

        video = video.astype("float32")

        if self.transform_names is not None:
            transform_names_list = self.transform_names.split()
            for transform_name in transform_names_list:
                if transform_name == "percentile_norm":
                    video = percentile_norm(video)
                if transform_name == "arcsinh":
                    if self.arcsinh_cofactor is None:
                        raise ValueError("arcsinh_cofactor must be provided for arcsinh transform.")
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
