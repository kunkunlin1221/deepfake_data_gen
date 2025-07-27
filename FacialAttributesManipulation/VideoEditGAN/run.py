import subprocess
from pathlib import Path


def video2frame(video, frame_folder="tmp/frames"):
    frame_folder = Path(frame_folder)
    if not frame_folder.exists():
        frame_folder.mkdir(parent=True, exist_ok=True)

    subprocess.run("python", "scripts/vid2frame.py", "--pathIn", str(video), "--pathOut", str(frame_folder), shell=True)


def prepare_landmarks(frame_folder):
    subprocess.run("cd", "3DDFA_v2", "&&", "sh", "./build.sh", shell=True)
    subprocess.run("cp", "../scrpits/single_video_smooth.py", "./", shell=True)
    subprocess.run("python", "single_video_smooth.py", "-f", str(frame_folder), shell=True)


def transform_faces(frame_folder, out_folder="tmp/aligned"):
    subprocess.run("cd", "..", shell=True)
    subprocess.run(
        "python",
        "scripts/align_faces_parallel.py",
        "--num_threads",
        "1",
        "--root_path",
        str(frame_folder),
        "--output_path",
        str(out_folder),
        shell=True,
    )


def main(video_folder, out_folder):
    video_folder = Path(video_folder)
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)

    video_paths = video_folder.glob("*.mp4")
    for video_path in video_paths:
        video2frame(video_path, "tmp/frames")
