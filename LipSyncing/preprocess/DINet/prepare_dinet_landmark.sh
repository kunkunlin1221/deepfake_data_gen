for file in /data/disk1/deepfake_data_gen/processed/LipSyncing/HDTF-test/_real_data_fps25/*.mp4; do
	build/bin/FaceLandmarkVidMulti \
	-f "$file" \
	-out_dir /data/disk1/deepfake_data_gen/processed/LipSyncing/HDTF-test/_dinet_landmark_fps25 \
	-2Dfp
done

for file in /data/disk1/deepfake_data_gen/processed/LipSyncing/VoxCeleb2/_real_data_fps25/*.mp4; do
	build/bin/FaceLandmarkVidMulti \
	-f "$file" \
	-out_dir /data/disk1/deepfake_data_gen/processed/LipSyncing/VoxCeleb2/_dinet_landmark_fps25 \
	-2Dfp
done

for file in /data/disk1/deepfake_data_gen/processed/LipSyncing/LRS2/_real_data_fps25/*.mp4; do
	build/bin/FaceLandmarkVidMulti \
	-f "$file" \
	-out_dir /data/disk1/deepfake_data_gen/processed/LipSyncing/LRS2/_dinet_landmark_fps25 \
	-2Dfp
done