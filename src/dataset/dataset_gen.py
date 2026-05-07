import argparse
import pickle
import os
import logging
import zarr
import tifffile as tiff
import numpy as np
import math
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

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in PKL is not a dict")

        if "Metadata" not in item:
            raise ValueError(f"Item {i} missing 'Metadata' key")

        if "Path" not in item and "Paths" not in item:
            raise ValueError(f"Item {i} must contain either 'Path' or 'Paths' key")

        if "Split_channels" not in item:
            raise ValueError(f"Item {i} missing 'Split_channels' key")

    return data


def extract_video(channel_mapping):
    if not channel_mapping["Split_channels"]:
        path = channel_mapping["Path"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        video = load_tiff(path)

    else:
        paths = channel_mapping["Paths"]

        channels = []
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            channel = load_tiff(path)

            if channel.shape[1] != 1:
                raise ValueError(f"Expected single channel in {
                                 path}, but got shape {channel.shape}")

            channels.append(channel)

        video = np.concatenate(channels, axis=1)

    return video


def throw_out_frames(video, metadata, acq_freq):
    T, C, H, W = video.shape
    original_acq_freq = metadata["Acq_freq_min"]
    step = acq_freq / original_acq_freq

    if not math.isclose(step, round(step), abs_tol=1e-5):
        raise ValueError(f"Acq_freq {acq_freq} is not an integer multiple of video Acq_freq_min {
                         original_acq_freq}")

    step = int(round(step))

    video = video[::step]
    metadata["Acq_freq_min"] = acq_freq
    metadata.pop("Acquisition_frequency_min", None)  # Remove redundancy

    return video, metadata


def create_zarr_dataset(
        items,
        out_path,
        acq_freq
):
    root = zarr.open(out_path, mode='w')

    compressors = zarr.codecs.BloscCodec(
        cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    idx = 0
    for item in tqdm(items):
        meta = item["Metadata"]

        try:
            video = extract_video(item)
            video, meta = throw_out_frames(video, meta, acq_freq)
        except Exception as e:
            logging.warning(f"Skipping item {idx} due to error: {e}")
            continue

        arr = root.create_array(
            name=f"{idx}",
            data=video,
            chunks=(32, 1, 224, 224),
            compressors=compressors,
        )
        idx += 1

        arr.attrs["Metadata"] = meta


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

    parser.add_argument(
        "--log",
        default="gen.log",
        help="Log file to save warnings and info about the matching process"
    )

    parser.add_argument(
        "--acq_freq",
        type=float,
        default=15.0,
        help="Target acquisition frequency in minutes (default: 1.0)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    np.random.seed(args.seed)

    try:
        items = load_pkl(args.input)
    except Exception as e:
        logging.error(f"Failed to load PKL file: {e}")
        return

    logging.info(f"Loaded {len(items)} items from PKL.")

    create_zarr_dataset(items, args.output, args.acq_freq)

    logging.info(f"Zarr dataset created at {args.output}")


if __name__ == "__main__":
    main()
