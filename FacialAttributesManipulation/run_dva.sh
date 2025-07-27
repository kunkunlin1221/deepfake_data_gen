export CMD="python run.py $1 $2"
# echo $CMD
# docker run \
#     --gpus all \
#     --shm-size=32g \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     --ipc=host \
#     --net=host \
#     -v $HOME/.cache:/root/.cache \
#     -v $(pwd)/Diffusion-Video-Autoencoders:/code \
#     -it --rm dva bash

docker run \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v $HOME/.cache:/root/.cache \
    -v $(pwd)/Diffusion-Video-Autoencoders:/code \
    -v /data:/data \
    -it --rm dva bash -c "$CMD"