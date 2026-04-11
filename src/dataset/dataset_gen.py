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

    return data


def extract_video(item, c1, c2):
    if "All_channels" in item:
        path = item["All_channels"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        video = load_tiff(path)

        try:
            c1_idx = item[c1]
            c2_idx = item[c2]
        except KeyError as e:
            raise ValueError(f"Missing channel key in item metadata: {e}")

        video = video[:, [c1_idx, c2_idx], :, :]

    else:
        channels = []

        try:
            tiff_paths = [item[c1], item[c2]]
        except KeyError as e:
            raise ValueError(f"Missing channel key in item metadata: {e}")

        for tiff_path in tiff_paths:
            if not os.path.exists(tiff_path):
                raise FileNotFoundError(f"Missing file: {tiff_path}")

            channel = load_tiff(tiff_path)

            if channel.shape[1] != 1:
                raise ValueError(f"Expected single channel in {
                                 tiff_path}, but got shape {channel.shape}")

            channels.append(channel)

        video = np.concatenate(channels, axis=1)

    return video


def downsample_video_2x(video):
    # Average 2x2 blocks to downsample by a factor of 2 in height and width
    return (
        video[:, :, 0::2, 0::2] +
        video[:, :, 0::2, 1::2] +
        video[:, :, 1::2, 0::2] +
        video[:, :, 1::2, 1::2]
    ) / 4.0


def clip_video(video, magnification, original_acq_freq, clip_frames, acq_freq, clip_size, clips_per_video):
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

    T, C, H, W = video.shape
    clips = []

    # Total frames needed to get clip_frames with the given step
    min_frames_needed = clip_frames * step - (step - 1)
    if T < min_frames_needed or H < clip_size or W < clip_size:
        raise ValueError(f"Video shape {video.shape} is smaller than clip requirements: "
                         f"clip_frames={clip_frames}, clip_size={clip_size}")

    for _ in range(clips_per_video):
        start_t = np.random.randint(0, T - min_frames_needed + 1)
        start_h = np.random.randint(0, H - clip_size + 1)
        start_w = np.random.randint(0, W - clip_size + 1)

        clip = video[start_t:start_t + min_frames_needed:step, :,
                     start_h:start_h + clip_size, start_w:start_w + clip_size]

        if downsample_2x:
            clip = downsample_video_2x(clip)

        clips.append(clip)

    # (clips_per_video, clip_frames, C, clip_size, clip_size)
    clips = np.stack(clips, axis=0)

    return clips


def create_zarr_dataset(
        items,
        out_path,
        clip_frames,
        acq_freq,
        clip_size,
        clips_per_video,
        c1,
        c2
):
    root = zarr.open(out_path, mode='w')

    compressors = zarr.codecs.BloscCodec(
        cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    root.create_array(
        name="Data",
        shape=(0, clip_frames, 2, clip_size, clip_size),
        chunks=(1, clip_frames, 2, clip_size, clip_size),
        dtype="uint16",
        compressors=compressors,
    )

    all_meta = []
    for i, item in enumerate(tqdm(items)):
        meta = item["Metadata"]
        # Original acquisition frequency in minutes
        original_acq_freq = meta["Acq_freq_min"]
        magnification = meta["Magnification"]
        meta.update({
            "C1": c1,
            "C2": c2,
            "Acq_freq_min": acq_freq,  # Update to target acquisition frequency
        })

        try:
            video = extract_video(item, c1, c2)
            clips = clip_video(video, magnification, original_acq_freq,
                               clip_frames, acq_freq, clip_size, clips_per_video)
        except Exception as e:
            logging.warning(f"Skipping item {i} due to error: {e}")
            continue

        root["Data"].append(clips, axis=0)
        all_meta.extend([meta] * clips.shape[0])  # Repeat meta for each clip

    root.attrs["Metadata"] = all_meta


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
        "--clip_frames",
        type=int,
        default=16,
        help="Number of frames per clip for Acq_freq_min=5 (default: 16)"
    )

    parser.add_argument(
        "--clip_size",
        type=int,
        default=224,
        help="Height and width of output clips for Magnification=20 (default: 224)"
    )

    parser.add_argument(
        "--clips_per_video",
        type=int,
        default=16,
        help="Number of clips to extract per video (default: 16)"
    )

    parser.add_argument(
        "--acq_freq",
        type=int,
        default=30,
        help="Acquisition frequency in minutes (default: 30)"
    )

    parser.add_argument(
        "--c1",
        type=str,
        default="Ch_H2B",
        help="Biosensor in channel 1 (default: Ch_H2B)"
    )

    parser.add_argument(
        "--c2",
        type=str,
        default="Ch_ERK-KTR",
        help="Biosensor in channel 2 (default: Ch_ERK-KTR)"
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

    create_zarr_dataset(items, args.output, args.clip_frames, args.acq_freq,
                        args.clip_size, args.clips_per_video, args.c1, args.c2)

    logging.info(f"Zarr dataset created at {args.output}")


if __name__ == "__main__":
    main()
