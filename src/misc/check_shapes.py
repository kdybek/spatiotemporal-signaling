import tifffile
from pathlib import Path

if __name__ == "__main__":
    tiff_dir = Path("tiff_symlinks")
    sizes_counter = {}
    counter = 0
    for tiff_file in tiff_dir.glob("*.tiff"):
        image = tifffile.imread(tiff_file)
        shape = image.shape
        sizes_counter[shape] = sizes_counter.get(shape, 0) + 1
        if counter % 100 == 0:
            print(f"---CHECKPOINT {counter // 100}---")
            for shape, count in sizes_counter.items():
                print(f"Shape: {shape}, Count: {count}")
        counter += 1

    print("---Unique shapes and their counts---")
    for shape, count in sizes_counter.items():
        print(f"Shape: {shape}, Count: {count}")