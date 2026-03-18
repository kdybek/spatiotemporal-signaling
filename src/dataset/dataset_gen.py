import argparse
import pickle
import os

import zarr
import tifffile as tiff
import numpy as np
from tqdm import tqdm


def load_tiff(path):
    arr = tiff.imread(path)

    # Normalize to (T, C, H, W)
    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        arr = arr[:, None, ...]
    elif arr.ndim == 4:
        # detect channel position
        if arr.shape[-1] <= 10:
            arr = np.moveaxis(arr, -1, 1)

    return arr


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError("PKL must contain a list of dicts")

    for i, item in enumerate(data):
        if "tiff_path" not in item or "metadata" not in item:
            raise ValueError(f"Item {i} missing 'path' or 'metadata'")

    return data


def create_zarr_dataset(items, out_path):
    root = zarr.open(out_path, mode='w')
    root.attrs['num_videos'] = len(items)

    for i, item in enumerate(tqdm(items)):
        path = item["tiff_path"]
        meta = item["metadata"]

        compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        video = load_tiff(path)
        T, C, H, W = video.shape

        group = root.create_group(f"video_{i:05d}")

        group.create_array(
            name="data",
            data=video,
            compressors=compressors
        )

        # attach metadata
        group.attrs.update(meta)
        group.attrs['shape'] = (T, C, H, W)

    # important for fast loading later
    zarr.consolidate_metadata(root.store)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PKL-listed TIFF videos to Zarr dataset"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to .pkl file containing list of {'path','metadata'} dicts"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output Zarr directory"
    )

    args = parser.parse_args()

    items = load_pkl(args.input)

    print(f"Loaded {len(items)} entries from PKL")

    create_zarr_dataset(items, args.output)


if __name__ == "__main__":
    main()
