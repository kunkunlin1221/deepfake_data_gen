rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/hdtf/hdtf_dataset/hdtf_test_data/test_videos/* /data/disk1/deepfake_data_gen/raw/HDTF-test

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/voxceleb-2/test_acc /data/disk1/deepfake_data_gen/raw/VoxCeleb2
rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/voxceleb-2/test_mp4 /data/disk1/deepfake_data_gen/raw/VoxCeleb2

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/LRS2/mvlrs_v1/main /data/disk1/deepfake_data_gen/raw/LRS2

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/rovi/data/* /data/disk1/deepfake_data_gen/raw/ROVI

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/davis2018/content_deletion_masks /data/disk1/deepfake_data_gen/raw/DAVIS
rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/davis2018/DAVIS_trainval /data/disk1/deepfake_data_gen/raw/DAVIS

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/youtube-vos/youtube-vos-2018/content_deletion_mask /data/disk1/deepfake_data_gen/raw/YOUTUBE-VOS
rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/youtube-vos/youtube-vos-2018/test_all_frames /data/disk1/deepfake_data_gen/raw/YOUTUBE-VOS

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/panda-70m/panda70m_human_highquality /data/disk1/deepfake_data_gen/raw/Panda70M

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/balancecc-human/BalanceCC/BalanceCC_human_videos /data/disk1/deepfake_data_gen/raw/BalanceCC

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/balancecc-human/BalanceCC/BalanceCC_human_videos /data/disk1/deepfake_data_gen/raw/BalanceCC
