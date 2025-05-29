from pathlib import Path
from shutil import copyfile

import numpy as np
from fire import Fire

np.random.seed(42)


def main(
    src_folder: str = "/data/disk1/deepfake_data_gen/raw/HDTF-test",
    dst_folder: str = "/data/disk1/deepfake_data_gen/processed/LipSyncing/HDTF-test/_real_data",
    n_samples: int = 150,
):
    """
    List all MP4 files in the specified folder and print their paths.

    Args:
        src_folder (str): Path to the folder containing MP4 files.
    """
    src_folder = Path(src_folder)
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder {src_folder} does not exist.")

    dst_folder = Path(dst_folder)
    if not dst_folder.exists():
        print(f"Destination folder {dst_folder} does not exist. Creating it.")
        dst_folder.mkdir(parents=True, exist_ok=True)

    mp4_files = list(src_folder.glob("*.mp4"))
    for mp4_file in mp4_files:
        dst_path = dst_folder / mp4_file.name
        copyfile(mp4_file, dst_path)


Fire(main)
