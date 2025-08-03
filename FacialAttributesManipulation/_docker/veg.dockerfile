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
	libharfbuzz-dev libfribidi-dev libxcb1-dev openssl libffi-dev \
	libssl-dev libedit-dev libncurses5-dev libreadline-dev xz-utils \
	ca-certificates

RUN unattended-upgrade && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# install miniconda (comes with python 3.10 default)
ARG MINICONDA_PREFIX=/home/user/miniconda3

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
	/bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX &&\
	$MINICONDA_PREFIX/bin/conda clean -ya &&\
	$MINICONDA_PREFIX/bin/conda install -y python=3.6

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

COPY veg_requirements.txt /tmp/veg_requirements.txt
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r /tmp/veg_requirements.txt

# Clone and build the project (inside the activated env)
RUN git clone https://github.com/cleardusk/3DDFA_V2.git && \
	cd 3DDFA_V2 && \
	sh ./build.sh

# RUN echo "export TORCH_CUDA_ARCH_LIST=8.0" > ~/.bashrc

# Working directory
WORKDIR /code
# Set default entrypoint (interactive shell in veg)
CMD ["bash"]