import pandas as pd
from pathlib import Path
import re
import logging


SOURCE_DIRS = [
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Chemotherapy",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_RSK",
    "/mnt/imaging.data/pgagliardi/MDCK_TimeLapse",
]


def find_metadata(exp_df, tiff_filename, tiff_dir_path):
    patters1 = re.compile(r"^(\d{2})(?:_Ori)?\.tiff?$")
    if match := patters1.match(tiff_filename):
        position = int(match.group(1))
        row = exp_df[exp_df["Position"] == position]
        if not row.empty:
            return row.iloc[0].to_dict()
        else:
            logging.warning(f"No matching metadata found: {tiff_filename} in {tiff_dir_path}.")
            return {}

    logging.warning(f"Unexpected TIFF filename format: {tiff_filename} in {tiff_dir_path}.")
    return {}


def get_data_from_dir(main_dir):
    root = Path(main_dir)

    for subdir in root.iterdir():
        if subdir.is_dir():
            exp_desc_file = subdir / "experimentDescription.csv"
            tiff_dir = subdir / "TIFFs"
            if exp_desc_file.exists():
                df = pd.read_csv(exp_desc_file, sep=None, engine="python")
                print(subdir, df.shape)

                for tiff_path in tiff_dir.glob("*.tif*"):
                    metadata = find_metadata(df, tiff_path.name, tiff_path)


if __name__ == "__main__":
    logging.basicConfig(
        filename='data_gen.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    for source_directory in SOURCE_DIRS:
        get_data_from_dir(source_directory)
