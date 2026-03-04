import zarr
import torch
import numpy as np


class N5VideoDataset(Dataset):
    def __init__(
        self,
        n5_path,
        dataset_key="clips",
        num_frames=16,
        transform=None
    ):
        self.store = zarr.N5Store(n5_path)
        self.root = zarr.open(self.store, mode="r")
        self.dataset_key = dataset_key
        self.num_frames = num_frames
        self.transform = transform

        # assume videos are stored as clips/<video_id>
        self.video_keys = list(self.root[self.dataset_key].keys())

    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        key = self.video_keys[idx]
        video = self.root[f"{self.dataset_key}/{key}"][:]  # (T,C,H,W)

        T = video.shape[0]

        # Random temporal sampling
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames).astype(int)
            video = video[indices]
        else:
            # pad if too short
            pad = self.num_frames - T
            pad_frames = np.repeat(video[-1:], pad, axis=0)
            video = np.concatenate([video, pad_frames], axis=0)

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        return video