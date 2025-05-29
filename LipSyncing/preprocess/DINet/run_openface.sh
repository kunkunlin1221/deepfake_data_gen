docker run \
    --gpus all \
	-v /data:/data \
	-v $(pwd):/home/openface-build/script \
	-it --rm --name openface algebr/openface:latest