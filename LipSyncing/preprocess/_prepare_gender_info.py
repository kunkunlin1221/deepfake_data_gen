from pathlib import Path

import capybara as cb
import facekit as fk
import numpy as np
from fire import Fire
from tqdm import tqdm

np.random.seed(42)


def main(src_folder: str):
    # face_service = fk.FaceService(enable_gender=True, gender_kwargs={"model_path": "lcnet050_v2.onnx"})
    # Ensure the destination folder exists
    src_folder = Path(src_folder)

    # Iterate through all files in the source folder
    mp4_files = sorted(src_folder.rglob("*.mp4"))
    out_folder = src_folder.parent / "_gender"
    out_folder.mkdir(exist_ok=True, parents=True)
    outs = []
    for file in tqdm(mp4_files, desc="Extract frame from MP4 files"):
        frames = cb.video2frames(file, frame_per_sec=1)
        fpath = out_folder / f"{file.stem}.png"
        cb.imwrite(frames[0], fpath)
        outs.append(f"{fpath.name}\n")

    with open(out_folder / "label.txt", "w") as f:
        f.writelines(outs)


if __name__ == "__main__":
    Fire(main)
