# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1\
	DEBIAN_FRONTEND=noninteractive\
	PYTHONWARNINGS="ignore"\
	TZ=Asia/Taipei

RUN apt update -y && apt install -y software-properties-common wget apt-utils patchelf git git-lfs libprotobuf-dev\
	protobuf-compiler cmake git bash curl libturbojpeg exiftool ffmpeg poppler-utils\
	libheif-dev	libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev\
	libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk\
	libharfbuzz-dev libfribidi-dev libxcb1-dev\
	unzip

RUN unattended-upgrade && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# install miniconda (comes with python 3.10 default)
ARG MINICONDA_PREFIX=/home/user/miniconda3

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
	/bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX &&\
	$MINICONDA_PREFIX/bin/conda clean -ya &&\
	$MINICONDA_PREFIX/bin/conda install -y python=3.6

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

RUN pip install -U pip
RUN pip install opencv_python==4.6.0.66 numpy==1.16.6 python_speech_features==0.6 resampy==0.2.2 scipy==1.5.4 tensorflow==1.15.2 gdown
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /models
RUN pip install gdown && gdown 1CkeEn7l3PuubuJIMWNjWpIrt0HDd_AB3 && unzip asserts.zip

WORKDIR /code
CMD bash