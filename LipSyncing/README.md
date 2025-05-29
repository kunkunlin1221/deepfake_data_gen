# Steps for lip syncing

## Preparation

### 0. Install requirements

```bash
cd preprocess
pip install -r requirements
```

### 1. Sample real data

```bash
python sample_hdtf-test.py
python sample_lrs2.py
python sample_voxceleb2.py
```

### 2. Prepare fake audios

```bash
python prepare_fake_audio.py
```

### 3. Prepare videos with FPS=25

```bash
bash prepare_fps25.sh
```

## Generate deepfake videos

### [DINet](https://github.com/MRzzm/DINet)

#### 1. Build the docker image

```bash
cd _docker
bash build_dinet.sh
cd -
```

#### 2. Prepare the facial landmarks

```bash
cd preprocess/DINet
bash run_openface.sh
# inside docker
bash script/prepare_dinet_landmark.sh
```

#### 3. Generate deepfake videos

```bash
bash run_dinet.sh
```

### [LatentSync](https://github.com/bytedance/LatentSync)

#### 1. Build the Docker image

```bash
cd _docker
bash build_latentsync.sh
cd -
```

#### 2. Generate deepfake videos (Need GPU)

```bash
bash run_latentsync.sh
```

### [MuseTalk](https://github.com/TMElyralab/MuseTalk)

#### 1. Build the Docker image

```bash
cd _docker
bash build_musetalk.sh
cd -
```

#### 2. Generate deepfake videos (Need GPU)

```bash
bash run_musetalk.sh
```
