import pandas as pd
from pathlib import Path
import re
import logging
import pickle


SOURCE_DIRS = [
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Chemotherapy",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_RSK",
    "/mnt/imaging.data/pgagliardi/MDCK_TimeLapse",
]


def find_metadata_position(exp_df, tiff_filename, tiff_full_path):
    pattern = re.compile(r"^(?:[Ss]eries_?)?(\d+)(?:_[Oo]ri)?\.tiff?$")  # Matches "01.tif", "01_Ori.tif", etc.

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
    return s[0] + str(int(s[1:])) if len(s) > 1 else s


def check_well_format(s):
    return isinstance(s, str) and re.match(r"^[A-Z]\d+$", s) is not None


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

    if not check_well_format(exp_df["Well"].iloc[0]):
        logging.warning(f"Unexpected 'Well' format in metadata: {tiff_full_path}.")
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
    if "Well" in exp_df.columns:
        return find_metadata_well(exp_df, tiff_filename, tiff_full_path)
    else:
        return find_metadata_position(exp_df, tiff_filename, tiff_full_path)


def get_data_from_dir(main_dir):
    root = Path(main_dir)
    data = []

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
            metadata = find_metadata(df, tiff_path.name, tiff_path)

            if metadata is not None:
                metadata["Tiff_path"] = str(tiff_path)
                data.append({
                    "tiff_path": str(tiff_path),
                    "metadata": metadata
                })

    return data


def main():
    logging.basicConfig(
        filename="matching.log",
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data = []
    for source_directory in SOURCE_DIRS:
        data.extend(get_data_from_dir(source_directory))

    logging.info(f"Matching completed. Total matched entries: {len(data)}.")

    with open("matched.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
