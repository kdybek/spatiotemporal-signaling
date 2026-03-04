import tifffile
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm


SAMPLES_PER_VIDEO = 8
CLIP_LEN = 32
CROP_W = 256
CROP_H = 256
CHECKPOINT_BATCH_SIZE = 2048
DATASET_NAME = "/mnt/imaging.data/zppmimuw/dataset.h5"
TIFF_SYMLINKS_DIR = "/mnt/imaging.data/zppmimuw/tiff_symlinks"


def check_shape(shape):
    return (len(shape) == 3) and \
        (shape[1] == shape[2]) and \
        (shape[0] >= CLIP_LEN) and \
        (shape[1] >= CROP_W) and \
        (shape[2] >= CROP_H)


def get_samples(video):
    T = video.shape[0]
    H = video.shape[1]
    W = video.shape[2]

    t_starts = np.random.randint(0, T - CLIP_LEN + 1, size=SAMPLES_PER_VIDEO)
    y_starts = np.random.randint(0, H - CROP_H + 1, size=SAMPLES_PER_VIDEO)
    x_starts = np.random.randint(0, W - CROP_W + 1, size=SAMPLES_PER_VIDEO)

    samples = np.empty((SAMPLES_PER_VIDEO, 1, CLIP_LEN, CROP_H, CROP_W), dtype=video.dtype)

    for i, (t, y, x) in enumerate(zip(t_starts, y_starts, x_starts)):
        samples[i] = video[np.newaxis, t:t+CLIP_LEN, y:y+CROP_H, x:x+CROP_W]

    return samples


def create_initial_dataset():
    with h5py.File(DATASET_NAME, "w") as f:
        f.create_dataset(
            "clips",
            shape=(0, 1, CLIP_LEN, CROP_H, CROP_W),
            maxshape=(None, 1, CLIP_LEN, CROP_H, CROP_W),
            dtype="float32",
            chunks=(1, 1, CLIP_LEN, CROP_H, CROP_W),
            compression="gzip"
        )

        f.attrs["next_index"] = 0


if __name__ == "__main__":
    assert CHECKPOINT_BATCH_SIZE % SAMPLES_PER_VIDEO == 0, "Batch size must be a multiple of samples per video."

    dataset_path = Path(DATASET_NAME)
    tiffs_path = Path(TIFF_SYMLINKS_DIR)

    if not dataset_path.exists():
        create_initial_dataset()

    with h5py.File(DATASET_NAME, "a") as f:
        num_tiffs = len(list(tiffs_path.glob("*.tiff")))
        start_index = f.attrs["next_index"]
        dset = f["clips"]

        sample_batch = np.empty((CHECKPOINT_BATCH_SIZE, 1, CLIP_LEN, CROP_H, CROP_W), dtype='float32')
        sample_batch_idx = 0

        for i in tqdm(range(start_index, num_tiffs)):
            tiff_file = tiffs_path / f"{i}.tiff"
            video = tifffile.imread(tiff_file).astype('float32')
            shape = video.shape

            if check_shape(shape):
                samples = get_samples(video)
                sample_batch[sample_batch_idx:sample_batch_idx+SAMPLES_PER_VIDEO] = samples
                sample_batch_idx += SAMPLES_PER_VIDEO
                if sample_batch_idx == CHECKPOINT_BATCH_SIZE:
                    dset.resize(dset.shape[0] + CHECKPOINT_BATCH_SIZE, axis=0)
                    dset[-CHECKPOINT_BATCH_SIZE:] = sample_batch
                    sample_batch_idx = 0
                    f.attrs["next_index"] = i + 1
                    f.flush()