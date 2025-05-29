# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1\
	DEBIAN_FRONTEND=noninteractive\
	PYTHONWARNINGS="ignore"\
	TZ=Asia/Taipei

RUN apt update -y && apt install -y software-properties-common wget apt-utils patchelf git git-lfs libprotobuf-dev\
	protobuf-compiler cmake git bash curl libturbojpeg exiftool ffmpeg poppler-utils\
	libheif-dev	libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev\
	libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk\
	libharfbuzz-dev libfribidi-dev libxcb1-dev

RUN unattended-upgrade && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# install miniconda (comes with python 3.10 default)
ARG MINICONDA_PREFIX=/home/user/miniconda3

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
	/bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX &&\
	$MINICONDA_PREFIX/bin/conda clean -ya &&\
	$MINICONDA_PREFIX/bin/conda install -y python=3.10

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

RUN pip install -U pip wheel
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install diffusers==0.32.2 transformers==4.48.0 decord==0.6.0 accelerate==0.26.1 einops==0.7.0 omegaconf==2.3.0\
	opencv-python==4.9.0.80 mediapipe==0.10.11 python_speech_features==0.6 librosa==0.10.1 scenedetect==0.6.1\
	ffmpeg-python==0.2.0 imageio==2.31.1 imageio-ffmpeg==0.5.1 lpips==0.1.4 face-alignment==1.4.1 gradio==5.24.0\
	huggingface-hub==0.30.2 numpy==1.26.4 kornia==0.8.0 insightface==0.7.3 onnxruntime-gpu==1.21.0

# Download the checkpoints required for inference from HuggingFace
RUN huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir /checkpoints &&\
	huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir /checkpoints

WORKDIR /code
CMD bash