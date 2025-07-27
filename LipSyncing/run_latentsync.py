import random
import subprocess
from pathlib import Path

from fire import Fire


def get_docker_cmd(video_path: str, audio_path: str, output_path: str) -> str:
    return (
        "docker run --gpus all "
        "-v $HOME/.cache:/root/.cache "
        "-v $(pwd)/LatentSync:/code "
        "-v /data:/data "
        "-it --rm latentsync "
        "python -m scripts.inference "
        "--unet_config_path=configs/unet/stage2.yaml "
        "--inference_ckpt_path=/checkpoints/latentsync_unet.pt "
        "--whisper_model_folder=/checkpoints/whisper "
        "--inference_steps=20 "
        "--guidance_scale=2.0 "
        f"--video_path={video_path} "
        f"--audio_path={audio_path} "
        f"--video_out_path={output_path}"
    )


def main(mother_dir, fake_audio_dir):
    video_paths = list(Path(mother_dir, "_real_data").glob("*.mp4"))
    male_audio_paths = list(Path(fake_audio_dir).rglob("**/male/*.wav"))
    female_audio_paths = list(Path(fake_audio_dir).rglob("**/female/*.wav"))

    with open(Path(mother_dir, "gender.txt"), "r") as f:
        gender_labels = [line.strip().split(" ") for line in f.readlines()]
        gender_labels = {k: int(v) for k, v in gender_labels}

    random.shuffle(male_audio_paths)
    random.shuffle(female_audio_paths)

    # Reorder audio paths
    result_folder = Path(mother_dir, "LatentSync")
    result_folder.mkdir(exist_ok=True)

    fails = []

    for video_path in video_paths:
        gender = gender_labels[video_path.name]
        if gender:
            audio_path = random.choice(male_audio_paths)
        else:
            audio_path = random.choice(female_audio_paths)

        output_path = result_folder / video_path.name
        cmd = get_docker_cmd(
            video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
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
