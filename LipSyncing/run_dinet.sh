# docker run \
#     --gpus all \
#     --shm-size=32g \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     --ipc=host \
#     --net=host \
#     -v $HOME/.cache:/root/.cache \
#     -v $(pwd)/DINet:/code \
#     -it --rm dinet bash

python run_dinet.py\
	--mother_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/HDTF-test"\
	--fake_audio_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/_fake_audio"

python run_dinet.py --mother_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/VoxCeleb2"\
	--fake_audio_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/_fake_audio"

python run_dinet.py\
	--mother_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/LRS2"\
	--fake_audio_dir "/data/disk1/deepfake_data_gen/processed/LipSyncing/_fake_audio"