from pathlib import Path
import pandas as pd
import re
import logging
import pickle
import argparse


def has_well_but_is_edge_case(exp_path):
    EDGE_CASES = [
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_RSK/2021-03-05_MCF10A-WT_ERKKTR-GEM_RSK-inhibitors-combinations_UOplusSL",
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs/2020-07-06_E545KandH1047R_Geminin_ERK_drugs",
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs/2020-07-10_E545KandH1047R_Geminin_ERK_drugs"
    ]

    exp_path_str = str(exp_path)

    if exp_path_str in EDGE_CASES:
        return True
    else:
        return False


def remove_leading_zeros_well(s):
    if isinstance(s, str) and len(s) > 1 and s[0].isalpha() and s[1:].isdigit():
        return s[0] + str(int(s[1:])) if len(s) > 1 else s
    else:
        return ""


def preprocess_exp_desc_df(df, exp_path):
    # Drop columns with empty names
    df = df.loc[:, df.columns.notna() & (df.columns != "")]

    # Drop rows where all values are empty (NaN or "")
    df = df.replace("", pd.NA).dropna(how="all")

    if "Well" not in df.columns or df["Well"].isna().all() or has_well_but_is_edge_case(exp_path):
        if "Site" in df.columns and not df.duplicated(subset=["Site"]).any():
            key = ["Site"]
        elif "Position" in df.columns and not df.duplicated(subset=["Position"]).any():
            key = ["Position"]
        else:
            raise ValueError("No key found for experiment description.")
    elif "Well" in df.columns:
        df["Well"] = df["Well"].apply(remove_leading_zeros_well)
        df = df[df["Well"] != ""]  # Drop rows where "Well" is empty after processing
        if not df.duplicated(subset=["Well"]).any():
            original_len = len(df)
            df = df.loc[df.index.repeat(10)]
            df["Site"] = list(range(10)) * original_len
            df["Site_added"] = True
            key = ["Well", "Site"]
        elif "Site" in df.columns and not df.duplicated(subset=["Well", "Site"]).any():
            key = ["Well", "Site"]
        elif "Position" in df.columns and not df.duplicated(subset=["Well", "Position"]).any():
            key = ["Well", "Position"]
        else:
            raise ValueError("No key found for experiment description.")
    else:
        raise ValueError("No key found for experiment description.")

    return df, key


def preprocess_tiff_pos(tiff_path):
    # Matches "01.tif", "01_Ori.tif", etc.
    pattern = re.compile(r"^(?:C\d_?)?(?:[Ss]eries_?)?(\d+)(?:_[Oo]ri)?\.tiff?$")
    tiff_filename = tiff_path.name

    if match := pattern.match(tiff_filename):
        position = int(match.group(1))
    else:
        raise ValueError(f"Unexpected TIFF filename format: {tiff_path}.")

    return {
        "Path": tiff_path,
        "Desc": (position,),
    }


def preprocess_tiff_well(tiff_path, just_well):
    # For example, "WellA2_Site1.tiff"
    pattern1 = re.compile(r"^Well([A-Z]\d+).*Site(\d+).*\.tiff?$")
    # For example, "WellA2_Seq0000_A2_0001_WF-640.tiff"
    pattern2 = re.compile(r"^Well([A-Z]\d+)_Seq\d+_[A-Z]\d+_(\d+).*\.tiff?$")
    tiff_filename = tiff_path.name

    if match := pattern1.match(tiff_filename):
        well = match.group(1)
        site = int(match.group(2))
    elif match := pattern2.match(tiff_filename):
        well = match.group(1)
        site = int(match.group(2))
    else:
        raise ValueError(f"Unexpected TIFF filename format: {tiff_path}.")

    well = remove_leading_zeros_well(well)

    desc = (well,) if just_well else (well, site)

    return {
        "Path": tiff_path,
        "Desc": desc,
    }


def preprocess_tiff(tiff_path, key):
    if key == ["Site"] or key == ["Position"]:
        return preprocess_tiff_pos(tiff_path)
    elif key == ["Well"]:
        return preprocess_tiff_well(tiff_path, just_well=True)
    elif key == ["Well", "Site"] or key == ["Well", "Position"]:
        return preprocess_tiff_well(tiff_path, just_well=False)
    else:
        raise ValueError(f"Unsupported key: {key}")


def process_split_channel_matched_tiffs(tiff_paths, exp_metadata):
    channels = [k for k in exp_metadata if k.startswith("Ch_")]

    channel_metadata = {}
    channel_mapping = {}

    for channel in channels:
        channel_pattern = exp_metadata[channel]
        # Possible edge case: channel C1 mathing with WellC1
        channel_pattern_regex = rf"(?<!Well){re.escape(channel_pattern)}"
        channel_tiff_paths = [tiff_path for tiff_path in tiff_paths if re.search(
            channel_pattern_regex, str(tiff_path)) is not None]

        if len(channel_tiff_paths) == 0:
            raise ValueError(f"No TIFF found matching pattern '{
                             channel_pattern}' for channel {channel}.")
        elif len(channel_tiff_paths) > 1:
            raise ValueError(f"Multiple TIFFs found matching pattern '{
                             channel_pattern}' for channel {channel}: {channel_tiff_paths}.")

        channel_metadata[f"{channel}_tiff"] = str(channel_tiff_paths[0])
        channel_mapping[channel] = str(channel_tiff_paths[0])

    return channel_mapping, channel_metadata


def process_non_split_channel_matched_tiffs(tiff_paths, exp_metadata):
    if len(tiff_paths) != 1:
        raise ValueError(
            f"Expected exactly one TIFF path for non-split-channel experiment, but got {tiff_paths}.")

    channel_mapping = {"All_channels": str(tiff_paths[0])}

    channels = [k for k in exp_metadata if k.startswith("Ch_")]

    channel_metadata = {}

    for channel in channels:
        channel_num = exp_metadata[channel]
        channel_metadata[f"{channel}_tiff_idx"] = channel_num - 1
        channel_mapping[channel] = channel_num - 1

    channel_metadata["All_channels_tiff"] = str(tiff_paths[0])

    return channel_mapping, channel_metadata


def process_matched_tiffs(tiff_paths, exp_metadata):
    if exp_metadata.get("Split_channels", "F") == "T":
        return process_split_channel_matched_tiffs(tiff_paths, exp_metadata)
    else:
        return process_non_split_channel_matched_tiffs(tiff_paths, exp_metadata)


def get_data_from_exp(exp_path, exp_metadata):
    data = []

    exp_desc_file = exp_path / "experimentDescription.csv"
    tiff_dir = exp_path / "TIFFs"

    if not tiff_dir.exists():
        raise FileNotFoundError("TIFF directory not found.")

    if not exp_desc_file.exists():
        raise FileNotFoundError("Experiment description file not found.")

    exp_desc_df = pd.read_csv(exp_desc_file, sep=None, engine="python")

    exp_desc_df, exp_desc_df_key = preprocess_exp_desc_df(exp_desc_df, exp_path)

    tiffs = []
    for tiff_path in tiff_dir.glob("*.tif*"):
        tiffs.append(preprocess_tiff(tiff_path, exp_desc_df_key))

    for _, row in exp_desc_df.iterrows():
        row_desc = tuple(row[c] for c in exp_desc_df_key)
        matching_tiff_paths = [tiff["Path"]
                               for tiff in tiffs if tiff["Desc"] == row_desc]

        if len(matching_tiff_paths) == 0:
            if row.get("Site_added", False):  # If I added "Site" to the description it can not match to anything
                continue
            else:
                logging.warning(f"No TIFF found matching description {
                                row_desc} for experiment at {exp_path}.")
                continue

        try:
            channel_mapping, channel_metadata = process_matched_tiffs(
                matching_tiff_paths, exp_metadata)
        except Exception as e:
            logging.warning(f"Error processing matched TIFFs for description {
                            row_desc} in experiment at {exp_path}: {e}")
            continue

        metadata = row.dropna().to_dict()
        metadata.update(exp_metadata)
        metadata.update(channel_metadata)

        entry = {"Metadata": metadata}
        entry.update(channel_mapping)
        data.append(entry)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Match TIFF file paths to their corresponding metadata from experiment description CSVs, and save the results in a PKL file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV file containing list of experiments with 'Path' and metadata columns"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output PKL file to save list of {'tiff_path','metadata'} dicts"
    )

    parser.add_argument(
        "--log",
        default="matching.log",
        help="Log file to save warnings and info about the matching process"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    all_data = []
    exps_df = pd.read_csv(args.input).dropna(subset=["Path"])
    for _, row in exps_df[exps_df["Usable"] == "T"].iterrows():
        exp_path = Path(row["Path"])
        exp_metadata = row.copy().replace("", pd.NA).dropna().to_dict()
        try:
            all_data.extend(get_data_from_exp(exp_path, exp_metadata))
        except Exception as e:
            logging.warning(f"Error processing experiment at {exp_path}: {e}")

    logging.info(f"Matching completed. Total matched entries: {len(all_data)}.")

    with open(args.output, "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    main()
