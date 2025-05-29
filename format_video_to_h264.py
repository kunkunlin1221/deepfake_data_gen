import subprocess
from pathlib import Path
from shutil import copyfile

from fire import Fire


def convert_to_h264(video_path: str, output_path: str):
    subprocess.call(["ffmpeg", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", output_path, "-y"])


def main(folder: str):
    videos = Path(folder).rglob("**/*.mp4")
    for video_fpath in videos:
        video_fpath = Path(video_fpath)
        convert_to_h264(video_fpath, "tmp.mp4")
        copyfile("tmp.mp4", video_fpath)


if __name__ == "__main__":
    Fire(main)
