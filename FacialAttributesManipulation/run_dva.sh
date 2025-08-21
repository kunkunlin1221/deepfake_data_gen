docker run \
    --gpus all \
    -v $HOME/.cache:/root/.cache \
    -v $(pwd)/Diffusion-Video-Autoencoders:/code \
    -v /data:/data \
    -it --rm dva python run.py --src_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/VFHQ/real_data --dst_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/VFHQ/dva

docker run \
    --gpus all \
    -v $HOME/.cache:/root/.cache \
    -v $(pwd)/Diffusion-Video-Autoencoders:/code \
    -v /data:/data \
    -it --rm dva python run.py --src_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/CelebV-Text/real_data --dst_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/CelebV-Text/dva

docker run \
    --gpus all \
    -v $HOME/.cache:/root/.cache \
    -v $(pwd)/Diffusion-Video-Autoencoders:/code \
    -v /data:/data \
    -it --rm dva python run.py --src_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/HDTF/real_data --dst_folder /data/disk1/deepfake_gen/FacialAttributeManipulation/HDTF/dva