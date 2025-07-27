import random
import subprocess
from pathlib import Path

import yaml
from fire import Fire


def prepare_inference_config(video_path: str, audio_path: str, yaml_path: str) -> str:
    yaml_content = {
        "task_0": {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
        }
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def get_docker_cmd(inference_config: str, result_dir: str) -> str:
    return (
        "docker run --gpus all "
        "-v $HOME/.cache:/root/.cache "
        "-v $(pwd)/MuseTalk:/code "
        "-v /data:/data "
        "-it --rm musetalk "
        "python3 -m scripts.inference "
        "--unet_model_path=/models/musetalkV15/unet.pth "
        "--unet_config=/models/musetalkV15/musetalk.json "
        "--whisper_dir=/models/whisper "
        f"--inference_config={inference_config} "
        f"--result_dir={result_dir} "
        "--version=v15"
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
    result_folder = Path(mother_dir, "MuseTalk")
    result_folder.mkdir(exist_ok=True)
    fails = []

    for video_path in video_paths:
        gender = gender_labels[video_path.name]
        if gender:
            audio_path = random.choice(male_audio_paths)
        else:
            audio_path = random.choice(female_audio_paths)

        output_path = result_folder / video_path.name
        prepare_inference_config(video_path, audio_path, "MuseTalk/inference_config.yaml")
        cmd = get_docker_cmd(
            inference_config="inference_config.yaml",
            result_dir=str(output_path),
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
