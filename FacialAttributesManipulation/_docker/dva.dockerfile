# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1\
	DEBIAN_FRONTEND=noninteractive\
	PYTHONWARNINGS="ignore"\
	TZ=Asia/Taipei

# RUN apt update -y && apt install -y software-properties-common wget apt-utils patchelf git git-lfs libprotobuf-dev\
# 	protobuf-compiler cmake git bash curl libturbojpeg exiftool ffmpeg poppler-utils\
# 	libheif-dev	libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev\
# 	libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk\
# 	libharfbuzz-dev libfribidi-dev libxcb1-dev\
# 	libavformat-dev libavcodec-dev libavdevice-dev \
# 	libavutil-dev libavfilter-dev libswscale-dev libswresample-dev pkg-config

RUN apt-get update && apt-get install -y \
  apt-utils bash cmake curl exiftool ffmpeg git git-lfs \
  libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libfreetype6-dev libfribidi-dev \
  libharfbuzz-dev libheif-dev libjpeg8-dev liblcms2-dev libopenjp2-7-dev libprotobuf-dev \
  libswresample-dev libswscale-dev libtiff5-dev libturbojpeg libwebp-dev libxcb1-dev \
  pkg-config poppler-utils protobuf-compiler python3-tk patchelf software-properties-common \
  tcl8.6-dev tk8.6-dev wget zlib1g-dev

RUN unattended-upgrade && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# install miniconda (comes with python 3.10 default)
ARG MINICONDA_PREFIX=/home/user/miniconda3

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
	/bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX &&\
	$MINICONDA_PREFIX/bin/conda clean -ya &&\
	$MINICONDA_PREFIX/bin/conda install -y python=3.9

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

COPY dva_requirements.txt /tmp/dva_requirements.txt
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install -r /tmp/dva_requirements.txt
RUN pip install fire git+https://github.com/openai/CLIP.git

WORKDIR /code
CMD bash