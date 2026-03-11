import h5py
import numpy as np
from torch.utils.data import Dataset


class N5VideoDataset(Dataset):
    def __init__(
        self,
        h5_path,
        dataset_key="videos",
        num_frames=16,
    ):
        self.root = h5py.File(h5_path, "r")
        self.dataset_key = dataset_key
        self.num_frames = num_frames

    def __len__(self):
        return self.root[self.dataset_key].shape[0]

    def __getitem__(self, idx):
        video = self.root[self.dataset_key][idx, ...]  # (T,C,H,W)

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

        video = video.astype(np.float32) / 255.0  # Normalize to [0,1]

        return video
