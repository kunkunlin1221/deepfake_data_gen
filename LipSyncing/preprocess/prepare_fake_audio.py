from pathlib import Path

import numpy as np
from fire import Fire
from moviepy import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

np.random.seed(42)


def process_audio(mp4_file: str, mp3_file: str = "audio.mp3"):
    """
    Extract audio from a video file and save it as an MP3 file.

    Args:
            mp4_file (str): Path to the input video file.
            mp3_file (str): Path to the output audio file.
    """
    # Load the video clip
    video_clip = VideoFileClip(mp4_file)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    audio_clip.write_audiofile(mp3_file)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()


def main(
    src_folder: str = "/data/disk1/deepfake_data_gen/processed/LipSyncing",
    dst_folder: str = "/data/disk1/deepfake_data_gen/processed/LipSyncing/_fake_audio",
    low_seconds: int = 5,
    high_seconds: int = 15,
    n_samples: int = 1000,
):
    """
    Process all MP4 files in the source folder and save the extracted audio in the destination folder.

    Args:
        src_folder (str): Path to the source folder containing MP4 files.
        dst_folder (str): Path to the destination folder for saving audio files.
    """

    # Ensure the destination folder exists
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    if not src_folder.exists():
        raise FileNotFoundError(f"Source folder {src_folder} does not exist.")
    if not dst_folder.exists():
        print(f"Destination folder {dst_folder} does not exist. Creating it.")
        # Create the destination folder if it doesn't exist
        dst_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through all files in the source folder
    mp4_files = list(src_folder.rglob("**/_real_data/*.mp4"))
    chosen_mp4 = {}
    for file in tqdm(mp4_files, desc="Extract audio from MP4 files"):
        audio = AudioSegment.from_file(file)
        if len(audio) > 5 * 1000:  # 5 seconds
            chosen_mp4[file] = audio

    mp4_list = list(chosen_mp4.keys())
    for i in tqdm(range(n_samples), desc="Select random audio segments"):
        mp4_file = np.random.choice(mp4_list)
        audio = chosen_mp4[mp4_file]
        total_mili_seconds = len(audio)
        high = min(high_seconds, total_mili_seconds // 1000)
        selected_seconds = np.random.randint(low_seconds, high + 1)
        start_time = np.random.randint(0, total_mili_seconds - selected_seconds * 1000)
        end_time = start_time + selected_seconds * 1000
        selected_audio = audio[start_time:end_time]
        mp3_file = dst_folder / f"{i:05}_{mp4_file.stem}.mp3"
        wav_file = mp3_file.with_suffix(".wav")
        selected_audio.export(mp3_file, format="mp3")
        selected_audio.export(wav_file, format="wav")


if __name__ == "__main__":
    Fire(main)
