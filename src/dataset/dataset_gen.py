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
    if arr.ndim == 3:
        arr = arr[:, None, ...]
    elif arr.ndim == 4:
        # detect channel position
        if arr.shape[-1] <= 10:
            arr = np.moveaxis(arr, -1, 1)
    else:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} in file {path}")

    return arr


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError("PKL must contain a list of dicts")

    ALLOWED_KEYS = {"Metadata", "All_channels"} | {f"C{i}" for i in range(1, 10)}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in PKL is not a dict")

        if "Metadata" not in item:
            raise ValueError(f"Item {i} missing 'Metadata' key")

        if not item.keys() <= ALLOWED_KEYS:
            raise ValueError(f"Item {i} contains invalid keys: {
                             item.keys() - ALLOWED_KEYS}")

    return data


def extract_video(item):
    if "All_channels" in item:
        path = item["All_channels"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        video = load_tiff(path)
    else:
        channels = []
        # Sort keys to ensure C1, C2, ... order (and ignore non-channel keys).
        for key in sorted(item, key=lambda k: int(k[1:]) if k.startswith("C") else float('inf')):
            if key.startswith("C"):
                path = item[key]

                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing file: {path}")

                channel = load_tiff(path)

                if channel.shape[1] != 1:
                    raise ValueError(f"Expected single channel in {
                                     path}, but got shape {channel.shape}")

                channels.append(channel)

        video = np.concatenate(channels, axis=1)

    return video


def create_zarr_dataset(items, out_path):
    root = zarr.open(out_path, mode='w')
    root.attrs['num_videos'] = len(items)

    for i, item in enumerate(tqdm(items)):
        meta = item["Metadata"]
        video = extract_video(item)

        compressors = zarr.codecs.BloscCodec(
            cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

        T, C, H, W = video.shape

        group = root.create_group(f"video_{i:05d}")

        group.create_array(
            name="data",
            data=video,
            compressors=compressors,
            chunks=(16, C, H, W)
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
        help="Path to PKL file containing list of {'path','metadata'} dicts"
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
