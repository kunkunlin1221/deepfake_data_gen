import random
import subprocess
from pathlib import Path

from fire import Fire


def get_docker_cmd(
    video_path: str,
    audio_path: str,
    landmark_path: str,
    result_folder: str,
) -> str:
    return (
        "docker run --gpus all "
        "-v $HOME/.cache:/root/.cache "
        "-v $(pwd)/Diffusion-Video-Autoencoders:/code "
        "-v /data:/data "
        "-it --rm dva "
        "python edit_CLIP.py "
        "--mouth_region_size=256 "
        f"--source_video_path={video_path} "
        f"--source_openface_landmark_path={landmark_path} "
        f"--driving_audio_path={audio_path} "
        "--pretrained_clip_DINet_path=/models/asserts/clip_training_DINet_256mouth.pth "
        "--deepspeech_model_path=/models/asserts/output_graph.pb "
        f"--res_video_dir={result_folder}"
    )


def main(mother_dir, fake_audio_dir):
    video_paths = list(Path(mother_dir, "_real_data_fps25").glob("*.mp4"))
    male_audio_paths = list(Path(fake_audio_dir).rglob("**/male/*.wav"))
    female_audio_paths = list(Path(fake_audio_dir).rglob("**/female/*.wav"))

    with open(Path(mother_dir, "gender.txt"), "r") as f:
        gender_labels = [line.strip().split(" ") for line in f.readlines()]
        gender_labels = {k: int(v) for k, v in gender_labels}

    random.shuffle(male_audio_paths)
    random.shuffle(female_audio_paths)

    # Reorder audio paths
    result_folder = Path(mother_dir, "DINet")
    result_folder.mkdir(exist_ok=True)
    fails = []

    for video_path in video_paths:
        lmk_fpath = Path(mother_dir, "_dinet_landmark_fps25", video_path.stem + ".csv")
        gender = gender_labels[video_path.name]
        if gender:
            audio_path = random.choice(male_audio_paths)
        else:
            audio_path = random.choice(female_audio_paths)

        cmd = get_docker_cmd(
            video_path=str(video_path),
            audio_path=str(audio_path),
            landmark_path=str(lmk_fpath),
            result_folder=str(result_folder),
        )

        print(f"Running: {video_path.name} + {audio_path.name}")
        try:
            subprocess.run(cmd, shell=True)  # <== shell=True is needed for $HOME and $(pwd)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_path.name} with {audio_path.name}: {e}")
            fails.append(video_path)

    with open(result_folder / "fails.txt", "w") as f:
        for fail in fails:
            f.write(f"{fail.name}\n")


if __name__ == "__main__":
    Fire(main)
