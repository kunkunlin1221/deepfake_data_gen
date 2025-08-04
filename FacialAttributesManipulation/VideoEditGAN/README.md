# VideoEditGAN

## Prerequisites

### Download pretrained models

```bash
bash download_pretrained.sh
```

### Build docker

```bash
cd _docker
bash build_veg.sh
cd -
```

### Run Pipeline

#### 0. Prepare required materials

```bash
export VIDEO_PATH="examples/aamir_khan_clip.mp4"
export NAME="aamir_khan"
export OUTPUT_FOLDER="out/$NAME"
```

#### 1. Prepare frames

```bash
cd src
python scripts/vid2frame.py --pathIn $VIDEO_PATH --pathOut $FRAME_FOLDER/frames
```

#### 2. Get face landmarks

```bash
cd 3DDFA_V2
sh ./build.sh # if not built yet
cp ../scripts/single_video_smooth.py ./
python single_video_smooth.py -f ../$FRAME_FOLDER/frames
```

The `landmarks.npy` will be saved at `$OUTPUT_FOLDER/landmarks/landmarks.npy`.

#### 3. Align faces

```bash
cd ..
python scripts/align_faces_parallel.py --num_threads 1 --root_path $FRAME_FOLDER/frames --output_path $OUTPUT_FOLDER/aligned
```

#### 4. Unaligned faces
```bash
python scripts/unalign.py --ori_images_path $OUTPUT_FOLDER/frames --aligned_images_path $OUTPUT_FOLDER/aligned --output_path $OUTPUT_FOLDER/unaligned
```

#### 5. GAN inversion
```bash
cd PTI
python scripts/run_pti_multi.py --data_root ../$OUTPUT_FOLDER/aligned --run_name $NAME --checkpoint_path ../$OUTPUT_FOLDER/inverted
```

#### 6. Direct editing
```bash
python scripts/pti_styleclip.py --inverted_root ../$OUTPUT_FOLDER/inverted --run_name $NAME --aligned_frame_path ../$OUTPUT_FOLDER/aligned --output_root ../$OUTPUT_FOLDER/in_domain --use_multi_id_G
cd ..
```


#### 7. VEG flow-based method
```bash
python -W ignore scripts/temp_consist.py --edit_root $OUTPUT_FOLDER/in_domain --metadata_root $OUTPUT_FOLDER/unaligned --original_root $OUTPUT_FOLDER/frames --aligned_ori_frame_root $OUTPUT_FOLDER/aligned --checkpoint_path $OUTPUT_FOLDER/inverted --batch_size 1 --reg_frame 0.2 --weight_cycle 10.0 --weight_tv_flow 0.0 --lr 1e-3 --weight_photo 1.0 --reg_G 100.0 --lr_G 1e-04 --weight_out_mask 0.5 --weight_in_mask 0.0 --tune_w --epochs_w 10 --tune_G --epochs_G 3 --scale_factor 4 --in_domain --exp_name 'temp_consist' --run_name $NAME
```


#### 8. Unalignment
```bash
cd STIT
python video_stitching_tuning_ours.py --input_folder ../$OUTPUT_FOLDER/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/aligned_frames --output_folder ../$OUTPUT_FOLDER/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/aligned_frames/stitiched --edit_name 'eyeglasses' --latent_code_path ../$OUTPUT_FOLDER/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/variables.pth --gen_path ../$OUTPUT_FOLDER/in_domain/StyleCLIP/eyeglasses/temp_consist/tune_G/G.pth --metadata_path ../$OUTPUT_FOLDER/unaligned --output_frames --num_steps 50
```
