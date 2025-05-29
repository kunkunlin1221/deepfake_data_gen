import subprocess
from pathlib import Path

from fire import Fire


def convert_to_fps25(video_path: str, output_path: str):
    """
    Convert video to 25 FPS using ffmpeg.
    """
    subprocess.call(["ffmpeg", "-i", video_path, "-filter:v", "fps=25", output_path, "-y"])


def main(src_folder: str, dst_folder: str):
    """
    Convert videos to 25 FPS using ffmpeg.
    """
    videos = Path(src_folder).glob("*.mp4")
    for video_fpath in videos:
        video_fpath = Path(video_fpath)
        output_path = Path(dst_folder) / f"{video_fpath.stem}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_fps25(video_fpath, output_path)
        print(f"Converted {video_fpath} to {output_path}")


if __name__ == "__main__":
    Fire(main)
