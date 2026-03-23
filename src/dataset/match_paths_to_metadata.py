from pathlib import Path
import pandas as pd
import re
import logging
import pickle
import argparse


def has_well_but_is_edge_case(full_tiff_path):
    EDGE_CASES = [
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_RSK/2021-03-05_MCF10A-WT_ERKKTR-GEM_RSK-inhibitors-combinations_UOplusSL",
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs/2020-07-06_E545KandH1047R_Geminin_ERK_drugs",
        "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs/2020-07-10_E545KandH1047R_Geminin_ERK_drugs"
    ]

    exp_path_str = str(full_tiff_path.parent.parent)

    if exp_path_str in EDGE_CASES:
        return True
    else:
        return False


def find_metadata_position(exp_df, tiff_filename, tiff_full_path):
    pattern = re.compile(r"^(?:C\d_?)?(?:[Ss]eries_?)?(\d+)(?:_[Oo]ri)?\.tiff?$")  # Matches "01.tif", "01_Ori.tif", etc.

    if match := pattern.match(tiff_filename):
        position = int(match.group(1))
    else:
        logging.warning(f"Unexpected TIFF filename format: {tiff_full_path}.")
        return None

    if "Site" in exp_df.columns:
        row = exp_df[exp_df["Site"] == position]
    elif "Position" in exp_df.columns and not (exp_df["Position"] == 0).all():
        row = exp_df[exp_df["Position"] == position]
    else:
        logging.warning(f"Neither 'Site' nor 'Position' column found in metadata: {tiff_full_path}.")
        return None

    if len(row) == 1:
        return row.iloc[0].to_dict()
    elif row.empty:
        logging.warning(f"No matching metadata found for position {position} in {tiff_full_path}.")
        return None
    else:
        logging.warning(f"Multiple metadata entries found for position {position} in {tiff_full_path}.")
        return None


def remove_leading_zeros_well(s):
    if isinstance(s, str) and len(s) > 1 and s[0].isalpha() and s[1:].isdigit():
        return s[0] + str(int(s[1:])) if len(s) > 1 else s
    else:
        return "well, well, well..."  # Return a string that will not match any valid well name


def find_metadata_well(exp_df, tiff_filename, tiff_full_path):
    pattern1 = re.compile(r"^Well([A-Z]\d+).*Site(\d+).*\.tiff?$")  # For example, "WellA2_Site1.tiff"
    pattern2 = re.compile(r"^Well([A-Z]\d+)_Seq\d+_[A-Z]\d+_(\d+).*\.tiff?$")  # For example, "WellA2_Seq0000_A2_0001_WF-640.tiff"

    if match := pattern1.match(tiff_filename):
        well = match.group(1)
        site = int(match.group(2))
    elif match := pattern2.match(tiff_filename):
        well = match.group(1)
        site = int(match.group(2))
    else:
        logging.warning(f"Unexpected TIFF filename format: {tiff_full_path}.")
        return None

    rows = exp_df[exp_df["Well"].apply(remove_leading_zeros_well) == remove_leading_zeros_well(well)]

    if rows.empty:
        logging.warning(f"No matching metadata found for well {well} in {tiff_full_path}.")
        return None

    if len(rows) == 1:
        return rows.iloc[0].to_dict()
    elif "Site" in exp_df.columns:
        row = rows[rows["Site"] == site]
    elif "Position" in exp_df.columns and not (exp_df["Position"] == 0).all():
        row = rows[rows["Position"] == site]
    else:
        logging.warning(f"Multiple metadata entries found for well {well}, but was not able to resolve them: {tiff_full_path}.")
        return None

    if len(row) == 1:
        return row.iloc[0].to_dict()
    elif row.empty:
        logging.warning(f"No matching metadata found for well {well} and site/position {site} in {tiff_full_path}.")
        return None
    else:
        logging.warning(f"Multiple metadata entries found for well {well} and site/position {site} in {tiff_full_path}.")
        return None


def find_metadata(exp_df, tiff_filename, tiff_full_path):
    if "Well" not in exp_df.columns or exp_df["Well"].isna().all() or has_well_but_is_edge_case(tiff_full_path):
        return find_metadata_position(exp_df, tiff_filename, tiff_full_path)
    else:
        return find_metadata_well(exp_df, tiff_filename, tiff_full_path)


def get_all_channels(tiff_name, tiff_dir):
    base_name = re.sub(r"C[123]", "", tiff_name)
    all_channels = list(tiff_dir.glob(f"{base_name}"))
    return all_channels


def channel_type(tiff_name):
    if "C1" in tiff_name:
        return "C1"
    elif "C2" in tiff_name:
        return "C2"
    elif "C3" in tiff_name:
        return "C3"
    else:
        return "AllChannels"


def get_data_from_dir(main_dir, exp_metadata):
    root = Path(main_dir)
    data = []
    used = set()  # To track used TIFF paths and avoid duplicates, I need this to handle split channels

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        exp_desc_file = subdir / "experimentDescription.csv"
        tiff_dir = subdir / "TIFFs"

        if not tiff_dir.exists():
            logging.info(f"TIFF directory not found: {tiff_dir}. Skipping {subdir}.")
            continue

        if not exp_desc_file.exists():
            logging.info(f"Experiment description file not found: {exp_desc_file}. Skipping {subdir}.")
            continue

        df = pd.read_csv(exp_desc_file, sep=None, engine="python")

        for tiff_path in tiff_dir.glob("*.tif*"):
            if str(tiff_path) in used:
                continue

            metadata = find_metadata(df, tiff_path.name, tiff_path)
            metadata.update(exp_metadata)  # Add experiment-level metadata to the TIFF metadata

            if metadata is not None:
                all_channels = get_all_channels(tiff_path.name, tiff_dir)

                metadata["Experiment"] = subdir.name
                metadata["Tiff_path"] = str(tiff_path)

                entry = {"metadata": metadata}

                for channel_path in all_channels:
                    used.add(str(channel_path))
                    entry[channel_type(channel_path.name)] = str(channel_path)

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

    args = parser.parse_args()

    logging.basicConfig(
        filename="matching.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data = []
    data_df = pd.read_csv(args.input).dropna(subset=["Path"])
    for _, row in data_df[data_df["Usable"] == "T"].iterrows():
        source_directory = row["Path"]
        exp_metadata = row.drop(["Path"]).dropna().to_dict()
        data.extend(get_data_from_dir(source_directory, exp_metadata))

    logging.info(f"Matching completed. Total matched entries: {len(data)}.")

    with open(args.output, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
