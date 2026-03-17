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


def find_metadata(exp_df, tiff_filename, tiff_full_path):
    pattern1 = re.compile(r"^(\d+)(?:_Ori)?\.tiff?$")  # First checks for "01.tif" or "01_Ori.tif" etc.
    pattern2 = re.compile(r"^Well([A-Z]\d).*\.tiff?$")  # Then checks for "WellA2[...].tiff" etc.

    if match := pattern1.match(tiff_filename):
        position = int(match.group(1))
        col_name = "Position" if "Position" in exp_df.columns else "Site" if "Site" in exp_df.columns else None

        if col_name is None:
            logging.warning(f"Neither 'Position' nor 'Site' column found in experiment description: {tiff_full_path}.")
            return None

        row = exp_df[exp_df[col_name] == position]

        if not row.empty:
            if len(row) > 1:
                logging.warning(f"Multiple metadata entries found for position {position} in {tiff_full_path}.")

            return row.iloc[0].to_dict()
        else:
            logging.warning(f"No matching metadata found: {tiff_full_path}.")
            return None

    elif match := pattern2.match(tiff_filename):
        well_id = match.group(1)
        try:
            row = exp_df[exp_df["Well"] == well_id]
        except KeyError:
            logging.warning(f"'Well' column not found in experiment description: {tiff_full_path}.")
            return None

        if not row.empty:
            if len(row) > 1:
                logging.warning(f"Multiple metadata entries found for well {well_id} in {tiff_full_path}.")

            return row.iloc[0].to_dict()
        else:
            logging.warning(f"No matching metadata found: {tiff_full_path}.")
            return None

    logging.warning(f"Unexpected TIFF filename format: {tiff_full_path}.")
    return None


def get_data_from_dir(main_dir):
    root = Path(main_dir)
    data = []

    for subdir in root.iterdir():
        if subdir.is_dir():
            exp_desc_file = subdir / "experimentDescription.csv"
            tiff_dir = subdir / "TIFFs"

            if exp_desc_file.exists():
                df = pd.read_csv(exp_desc_file, sep=None, engine="python")

                for tiff_path in tiff_dir.glob("*.tif*"):
                    metadata = find_metadata(df, tiff_path.name, tiff_path)

                    if metadata is not None:
                        data.append({
                            "tiff_path": str(tiff_path),
                            "metadata": metadata
                        })

    return data


def main():
    logging.basicConfig(
        filename="matching.log",
        level=logging.INFO,
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
