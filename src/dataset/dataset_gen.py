import argparse
import pickle
import os
import logging
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


def choose_channels(video, item, c1, c2):
    channel1 = next((k for k, v in item if v == c1), None)
    channel2 = next((k for k, v in item if v == c2), None)

    if channel1 is None or channel2 is None:
        raise ValueError(f"Channels {c1} and/or {c2} not found in item metadata")

    idx1 = int(channel1[1:]) - 1
    idx2 = int(channel2[1:]) - 1

    if idx1 >= video.shape[1] or idx2 >= video.shape[1]:
        raise ValueError(f"Channel indices {
                         idx1} and/or {idx2} out of bounds for video with shape {video.shape}")

    return video[:, [idx1, idx2], :, :]


def downsample_video_2x(video):
    # Average 2x2 blocks to downsample by a factor of 2 in height and width
    return (
        video[:, :, 0::2, 0::2] +
        video[:, :, 0::2, 1::2] +
        video[:, :, 1::2, 0::2] +
        video[:, :, 1::2, 1::2]
    ) / 4.0


def clip_video(video, meta, clip_frames, clip_size, clips_per_video):
    if meta["Acq_freq_min"] != 5:
        raise ValueError(f"Expected Acq_freq_min=5, but got {
                         meta['Acq_freq_min']}")  # Don't handle for now

    if meta["Magnification"] == 40:
        downsample_2x = True
        clip_size *= 2  # We need to double the clip size to get the same field of view
    elif meta["Magnification"] == 20:
        downsample_2x = False
    else:
        raise ValueError(f"Unsupported Magnification {meta['Magnification']}")

    T, C, H, W = video.shape
    clips = []

    if T < clip_frames or H < clip_size or W < clip_size:
        raise ValueError(f"Video shape {video.shape} is smaller than clip requirements: "
                         f"clip_frames={clip_frames}, clip_size={clip_size}")

    for _ in range(clips_per_video):
        start_t = np.random.randint(0, T - clip_frames + 1)
        start_h = np.random.randint(0, H - clip_size + 1)
        start_w = np.random.randint(0, W - clip_size + 1)

        clip = video[start_t:start_t + clip_frames, :,
                     start_h:start_h + clip_size, start_w:start_w + clip_size]

        if downsample_2x:
            clip = downsample_video_2x(clip)

        clips.append(clip)

    # (clips_per_video, clip_frames, C, clip_size, clip_size)
    clips = np.stack(clips, axis=0)

    return clips


def append_to_zarr(root, clips, arr_name):
    current_len = root[arr_name].shape[0]
    new_len = current_len + clips.shape[0]

    root[arr_name].resize(new_len, axis=0)
    root[arr_name][current_len:new_len] = clips


def create_zarr_dataset(
        items,
        out_path,
        clip_frames,
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
        meta.update({
            "C1": c1,
            "C2": c2,
        })

        try:
            video = extract_video(item)
            video = choose_channels(video, item, c1, c2)
            clips = clip_video(video, clip_frames, clip_size, clips_per_video)
        except Exception as e:
            logging.warning(f"Skipping item {i} due to error: {e}")
            continue

        append_to_zarr(root, clips, "Data")
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
        help="Number of clips to extract per video (default: 4)"
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

    create_zarr_dataset(items, args.output, args.clip_frames,
                        args.clip_size, args.clips_per_video, args.c1, args.c2)

    logging.info(f"Zarr dataset created at {args.output}")


if __name__ == "__main__":
    main()
