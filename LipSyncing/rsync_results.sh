rsync -avzh --progress -e "ssh -i $HOME/.ssh/id_ed25519" data/processed/LipSyncing/* w-ssh-pp-jack-5a6ecf95aa7048418857fd0d2d9d2120@ssh.nvidia-oci.saturnenterprise.io:shared/pp/output/lip-syncing
