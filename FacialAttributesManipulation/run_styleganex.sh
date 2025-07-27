docker run \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v $HOME/.cache:/root/.cache \
    -v $(pwd)/StyleGANEX:/code \
    -it --rm styleganex bash