import os
import PIL
import dlib
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import argparse
import torch
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--workers", type=int, default=cpu_count())
args = parser.parse_args()

output_size = 256
transform_size = 4096
enable_padding = True
torch.backends.cudnn.benchmark = False
os.makedirs(args.output_dir, exist_ok=True)

# Collect image paths
img_files = [
    os.path.join(path, filename)
    for path, dirs, files in os.walk(args.root)
    for filename in files
    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
]
img_files.sort()


def process_image(index_imgfile):
    index, img_file = index_imgfile
    output_img = os.path.join(args.output_dir, f"{index:08}.png")
    if os.path.isfile(output_img):
        return

    try:
        img = dlib.load_rgb_image(img_file)
        dets = detector(img, 1)
        if len(dets) <= 0:
            print(f"[{index}] no face landmark detected")
            return

        shape = sp(img, dets[0])
        points = np.array([[shape.part(b).x, shape.part(b).y] for b in range(68)])
        lm = points

        # landmark regions
        lm_eye_left = lm[36:42]
        lm_eye_right = lm[42:48]
        lm_mouth_outer = lm[48:60]

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = Image.fromarray(img).convert("RGB")

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
            img = img.resize(rsize, Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

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
            pad_amt = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad_amt[1], pad_amt[3]), (pad_amt[0], pad_amt[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(x / pad_amt[0], (w - 1 - x) / pad_amt[2]),
                1.0 - np.minimum(y / pad_amt[1], (h - 1 - y) / pad_amt[3])
            )
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad_amt[:2]

        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

        img.save(output_img)

    except Exception as e:
        print(f"[{index}] Error: {e}")


def init_worker():
    global detector, sp
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if __name__ == "__main__":
    with Pool(args.workers, initializer=init_worker) as pool:
        list(tqdm(pool.imap_unordered(process_image, enumerate(img_files)), total=len(img_files)))
