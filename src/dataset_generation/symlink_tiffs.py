import os


LINK_DIR = "/mnt/imaging.data/zppmimuw/tiff_symlinks"
SOURCE_DIRS = [
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Chemotherapy",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_Geminin-Drugs",
    "/mnt/imaging.data/pgagliardi/MCF10A_TimeLapse_RSK",
    "/mnt/imaging.data/pgagliardi/MDCK_TimeLapse",
]
ITER = 0


def create_symlinks(src_dir, link_dir):
    """
    Traverse src_dir recursively and create symlinks for all .tiff files in link_dir.
    """
    global ITER
    
    if not os.path.exists(link_dir):
        os.makedirs(link_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.tiff'):
                src_path = os.path.join(root, file)
                link_path = os.path.join(link_dir, f"{ITER}.tiff")
                
                os.symlink(src_path, link_path)
                print(f"Linked: {src_path} -> {link_path}")
                ITER += 1

if __name__ == "__main__":
    for source_directory in SOURCE_DIRS:
        create_symlinks(source_directory, LINK_DIR)