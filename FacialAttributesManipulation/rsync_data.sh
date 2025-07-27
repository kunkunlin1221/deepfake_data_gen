rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/celebv-text/CelebV-Text/data/celebvtext_visual/* /data/disk1/deepfake_data_gen/raw/celebv-text

rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/vfhq-test/videos/* /data/disk1/deepfake_data_gen/raw/vfhq
