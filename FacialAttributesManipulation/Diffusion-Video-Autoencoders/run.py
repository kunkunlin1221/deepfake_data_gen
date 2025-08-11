# This file is used to deepfake videos by editing facial attributes using a diffusion model.
# The code is from editing_classifier.py and is designed to run in a Docker container.
import math
import os
from pathlib import Path

import dlib
import ffmpeg
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from fire import Fire
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import random

if True:
    import sys

    sys.path.append("src")
    from dataset import CelebAttrDataset, ImageDataset
    from experiment import LitModel
    from experiment_classifier import ClsModel
    from templates import diffusion_video_autoencoder
    from templates_cls import diffusion_video_autoencoder_cls


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert("RGBA")
    projected = img.convert("RGBA").transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)


def get_landmark(filepath, predictor, detector=None, fa=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if isinstance(filepath, Image.Image):
        img = np.array(filepath)
    else:
        img = dlib.load_rgb_image(filepath)
    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        break
    else:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        raise Exception(f"Did not detect any faces in image: {filepath}")
    # lm_chin = lm[0:17]  # left-right
    # lm_eyebrow_left = lm[17:22]  # left-right
    # lm_eyebrow_right = lm[22:27]  # left-right
    # lm_nose = lm[27:31]  # top-down
    # lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    # lm_mouth_inner = lm[60:68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y


def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, Image.Image):
        img = filepath
    else:
        img = Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]
    # Transform
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)
    return img


def video_to_pngs(video_path: str, frame_folder: str):
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    (
        ffmpeg.input(str(video_path))
        .output(os.path.join(frame_folder, "%05d.png"), format="image2", vcodec="png")
        .run(overwrite_output=True)
    )


cur_folder = Path(__file__).parent


def edit_video(
    src_video: str,
    dst_video: str,
    batch_size: int = 8,
    T: int = 100,
    max_num: int = None,
    attribute: str = "Blond_Hair",
    scale: float = 0.25,
    normalize: bool = True,
    gpus: list = [0],  # Specify the GPUs to use
    dif_ckpt_path: str = cur_folder / "checkpoints/diffusion_video_autoencoder/epoch=51-step=1000000.ckpt",
    cls_ckpt_path: str = cur_folder / "checkpoints/diffusion_video_autoencoder_cls/last.ckpt",
):
    gpus = [0]
    device = "cuda:0"

    # preparation
    src_video = Path(src_video)
    frame_folder = Path("tmp/sample_video")
    video_to_pngs(src_video, frame_folder)

    # load the diffusion model
    conf = diffusion_video_autoencoder(gpus)
    state = torch.load(dif_ckpt_path, map_location="cpu")
    model = LitModel(conf)
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # load the classifier
    cls_conf = diffusion_video_autoencoder_cls(gpus)
    cls_model = ClsModel(cls_conf)
    state = torch.load(cls_ckpt_path, map_location="cpu")
    print("latent step:", state["global_step"])
    cls_model.load_state_dict(state["state_dict"], strict=False)
    cls_model.to(device)
    # print(CelebAttrDataset.id_to_cls)

    # create log directory
    log_dir = Path("editing_classifier") / src_video.stem
    crop_dir = log_dir / "crop"
    recon_dir = log_dir / "recon"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    if not crop_dir.exists():
        crop_dir.mkdir(parents=True, exist_ok=True)
    if not recon_dir.exists():
        recon_dir.mkdir(parents=True, exist_ok=True)

    # load face landmarker and detector
    predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    images = []
    for fname in sorted(os.listdir(frame_folder)):
        path = os.path.join(frame_folder, fname)
        fname = fname.split(".")[0]
        images.append((fname, path))

    cs, xs, ys = [], [], []
    for _, path in images:
        c, x, y = compute_transform(path, predictor, detector=detector, scale=1.0)
        cs.append(c)
        xs.append(x)
        ys.append(y)
    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    cs = gaussian_filter1d(cs, sigma=1.0, axis=0)
    xs = gaussian_filter1d(xs, sigma=3.0, axis=0)
    ys = gaussian_filter1d(ys, sigma=3.0, axis=0)
    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)
    orig_images = []
    for quad, (_, path) in tqdm(zip(quads, images), total=len(quads)):
        crop = crop_image(path, 1024, quad.copy())
        crop.save(f"{log_dir}/crop/%s.jpg" % path.split("/")[-1].split(".")[0])  # , quality=100, subsampling=0)
        orig_image = Image.open(path)
        orig_images.append(orig_image)

    image_size = 256
    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]

    data = ImageDataset(
        f"{log_dir}/crop",
        image_size=conf.img_size,
        exts=["jpg", "JPG", "png"],
        do_augment=False,
        sort_names=True,
        max_num=max_num,
    )
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    conds = []
    for batch in tqdm(dataloader, desc="Forward cond"):
        imgs = batch["img"]
        indices = batch["index"]

        cond = model.ema_model.encoder.id_forward(imgs.to(device))
        conds.append(cond)

    cond = torch.cat(conds, dim=0)
    avg_cond = torch.mean(cond, dim=0, keepdim=True)  # .expand(len(x_start), -1)
    video_frames = []

    if normalize:
        avg_cond_norm = cls_model.normalize(avg_cond)

    for batch in tqdm(dataloader, desc="Compute reconstructions"):
        imgs = batch["img"]
        indices = batch["index"]

        with torch.no_grad():
            avg_latent = model.ema_model.encoder.forward_with_id(avg_cond.expand(len(imgs), -1), imgs.to(device))
            avg_xT = model.encode_stochastic(imgs.to(device), avg_latent, T=T)
            mask = model.ema_model.encoder.face_mask(imgs.to(device), for_video=True)
            avg_img_recon = model.render(avg_xT, avg_latent, T=T)

        ori = (imgs + 1) / 2
        for index in range(len(imgs)):
            file_name = data.paths[indices[index]]
            save_image(ori[index], f"{log_dir}/recon/orig_{file_name}")
            save_image(avg_xT[index], f"{log_dir}/recon/avg_xT_{file_name}")
            save_image(avg_img_recon[index], f"{log_dir}/recon/avg_recon_{file_name}")

        cls_id = CelebAttrDataset.cls_to_id[attribute]
        if normalize:
            avg_cond2 = avg_cond_norm + scale * math.sqrt(512) * F.normalize(
                cls_model.classifier.weight[cls_id][None, :], dim=1
            )
            avg_cond2 = l2_norm(cls_model.denormalize(avg_cond2))
        else:
            avg_cond2 = l2_norm(avg_cond + scale * math.sqrt(512) * cls_model.classifier.weight[cls_id][None, :])

        if not os.path.exists(f"{log_dir}/{attribute}_{scale:.2f}"):
            os.mkdir(f"{log_dir}/{attribute}_{scale:.2f}")

        avg_latent2 = model.ema_model.encoder.forward_with_id(avg_cond2.expand(len(imgs), -1), imgs.to(device))
        avg_img = model.render(avg_xT, avg_latent2, T=T)

        for index in range(len(imgs)):
            file_name = data.paths[indices[index]]
            # save_image(avg_img[index], f'{log_dir}/{attribute}_{scale:.2f}/avg_mani_{file_name}')
            paste_bg = avg_img[index].unsqueeze(0) * mask[index].unsqueeze(0) + (
                (imgs[index].to(device).unsqueeze(0) + 1) / 2 * (1 - mask[index].unsqueeze(0))
            )
            save_image(paste_bg[0], f"{log_dir}/{attribute}_{scale:.2f}/paste_avg_mani_{file_name}")
            paste_bg_crop = tensor2pil((paste_bg[0] * 2) - 1)
            paste_bg_pasted_image = paste_image(
                inverse_transforms[indices[index]], paste_bg_crop, orig_images[indices[index]]
            )
            paste_bg_pasted_image = paste_bg_pasted_image.convert("RGB")
            video_frames.append(paste_bg_pasted_image)
            paste_bg_pasted_image.save(f"{log_dir}/{attribute}_{scale:.2f}/paste_final_avg_mani_{file_name}")

        imageio.mimwrite(
            str(dst_video),
            video_frames,
            fps=20,
            output_params=["-vf", "fps=20"],
        )


def edit_videos(
    src_folder: str,
    dst_folder: str,
):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True, exist_ok=True)

    attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    for src_video in src_folder.glob("*.mp4"):
        attribute = random.choice(attributes)
        dst_video = dst_folder / (src_video.stem + f"_{attribute}.mp4")
        edit_video(
            src_video=str(src_video),
            dst_video=str(dst_video),
            batch_size=8,
            T=500,
            max_num=None,
            attribute=attribute,
            scale=0.25,
            normalize=True,
        )

if __name__ == "__main__":
    Fire(edit_videos)
