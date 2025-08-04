pip install gdown

mkdir -p src/PTI/pretrained_models

gdown 1CEO3eQr46KnfB8e-U8AZ9LDHaL0NwJda -O src/PTI/pretrained_models/eyeglasses.pt
gdown 1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm -O src/PTI/pretrained_models/e4e_ffhq_encode.pt
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl  -O src/PTI/pretrained_models/ffhq.pkl

mkdir -p src/pretrained_models

gdown --folder 1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT -O src/pretrained_models

cp src/PTI/pretrained_models/eyeglasses.pt src/pretrained_models

gdown 1cDvUHPTZQAEWvfiK9C0nDuI9C3Qdgbbp -O src/STIT/pretrained_models.zip
unzip -o src/STIT/pretrained_models.zip -d src/STIT
rm src/STIT/pretrained_models.zip