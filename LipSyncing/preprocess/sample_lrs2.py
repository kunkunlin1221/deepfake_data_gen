from pathlib import Path
from shutil import copyfile

import numpy as np
from fire import Fire

np.random.seed(42)


def main(
    src_folder: str = "/data/disk1/deepfake_data_gen/raw/LRS2/main",
    dst_folder: str = "/data/disk1/deepfake_data_gen/processed/LipSyncing/LRS2/_real_data",
    n_samples: int = 200,
):
    src_folder = Path(src_folder)
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder {src_folder} does not exist.")

    dst_folder = Path(dst_folder)
    if not dst_folder.exists():
        print(f"Destination folder {dst_folder} does not exist. Creating it.")
        dst_folder.mkdir(parents=True, exist_ok=True)
    samples = [x for x in src_folder.glob("*") if x.is_dir()][:n_samples]

    for sample in samples:
        videos = sorted(sample.glob("*.mp4"))
        copyfile(videos[0], dst_folder / f"{sample.name}_{videos[0].stem}.mp4")

    # folders = list(src_folder.glob("*"))
    # inds = np.random.randint(0, len(folders), size=n_samples)
    # chosen_mp4s = []
    # for i in inds:
    #     folder = folders[i]
    #     mp4_files = list(folder.rglob("**/*.mp4"))
    #     np.random.randint(0, len(mp4_files))
    #     mp4_file = mp4_files[0]
    #     chosen_mp4s.append(str(mp4_file))
    #     dst_name = "-".join(mp4_file.parts[-4:])
    #     dst_path = dst_folder / dst_name
    #     copyfile(mp4_file, dst_path)


Fire(main)
