# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

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
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install diffusers==0.30.2 accelerate==0.28.0 numpy==1.23.5 tensorflow==2.12.0 tensorboard==2.12.0\
	opencv-python==4.9.0.80 soundfile==0.12.1 transformers==4.39.2 huggingface_hub==0.30.2 librosa==0.11.0\
	einops==0.8.1 gradio==5.24.0\
	gdown requests imageio[ffmpeg]\
	omegaconf ffmpeg-python moviepy

RUN pip install --no-cache-dir -U openmim &&\
	mim install mmengine &&\
	mim install "mmcv==2.0.1" &&\
	mim install "mmdet==3.1.0" &&\
	mim install "mmpose==1.1.0"


RUN mkdir -p /models/musetalk /models/musetalkV15 /models/syncnet /models/dwpose /models/face-parse-bisent /models/sd-vae models/whisper

# Install required packages
RUN pip install -U "huggingface_hub[cli]" gdown

# Set HuggingFace mirror endpoint
ENV HF_ENDPOINT=https://hf-mirror.com
ENV CheckpointsDir=/models

# Download MuseTalk V1.0 weights
RUN huggingface-cli download TMElyralab/MuseTalk \
	--local-dir $CheckpointsDir \
	--include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
RUN huggingface-cli download TMElyralab/MuseTalk \
	--local-dir $CheckpointsDir \
	--include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
RUN huggingface-cli download stabilityai/sd-vae-ft-mse \
	--local-dir $CheckpointsDir/sd-vae \
	--include "config.json" "diffusion_pytorch_model.bin" "diffusion_pytorch_model.safetensors"

# Download Whisper weights
RUN huggingface-cli download openai/whisper-tiny \
	--local-dir $CheckpointsDir/whisper \
	--include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
RUN huggingface-cli download yzd-v/DWPose \
	--local-dir $CheckpointsDir/dwpose \
	--include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
RUN huggingface-cli download ByteDance/LatentSync \
	--local-dir $CheckpointsDir/syncnet \
	--include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
RUN gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth &&\
	curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

WORKDIR /code
CMD bash