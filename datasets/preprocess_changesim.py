import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from LightGlue.lightglue import SuperPoint, LightGlue
from LightGlue.lightglue.utils import rbd
from torchvision import transforms
from os.path import join as osp

from project_config import resolve_device

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def preprocess_tensor(img_np, device):
    img = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.
    return img.to(device)

def align_and_warp(t0_img_np, t1_img_np, matcher, extractor, device):
    h, w = t0_img_np.shape[:2]

    t0_tensor = preprocess_tensor(t0_img_np, device)
    t1_tensor = preprocess_tensor(t1_img_np, device)

    feats0 = extractor.extract(t0_tensor)[0]
    feats1 = extractor.extract(t1_tensor)[0]

    matches = matcher({'image0': feats0, 'image1': feats1})
    matches = rbd(matches)

    kpts0 = matches['keypoints0'].cpu().numpy()
    kpts1 = matches['keypoints1'].cpu().numpy()
    matches0 = matches['matches0'].cpu().numpy()

    valid = matches0 > -1
    if valid.sum() < 4:
        return t1_img_np  # return original if not enough matches

    src = kpts1[valid]
    dst = kpts0[matches0[valid]]

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(t1_img_np, H, (w, h))
    return warped

def preprocess_dataset(data_root, out_root, device="auto"):
    device = resolve_device(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    extractor = SuperPoint().eval().to(device)

    os.makedirs(out_root, exist_ok=True)

    for house_idx in range(6):
        seq_path = osp(data_root, f'Warehouse_{house_idx}', 'Seq_0')
        t0_dir = osp(seq_path, 'rgb')
        t1_dir = osp(seq_path, 't0', 'rgb')
        out_dir = osp(seq_path, 't0', 'rgb_aligned')

        if not os.path.exists(t0_dir) or not os.path.exists(t1_dir):
            continue

        os.makedirs(out_dir, exist_ok=True)

        filenames = sorted(os.listdir(t0_dir))

        for fname in tqdm(filenames, desc=f'Processing Warehouse_{house_idx}'):
            t0_img = load_image(osp(t0_dir, fname))
            t1_img = load_image(osp(t1_dir, fname))
            warped_t1 = align_and_warp(t0_img, t1_img, matcher, extractor, device)

            Image.fromarray(warped_t1).save(osp(out_dir, fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ChangeSim aligned RGB frames")
    parser.add_argument("--data_root", required=True, help="Root containing Warehouse_* training folders")
    parser.add_argument("--out_root", required=True, help="Output root for warped data")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:N, or cpu")
    args = parser.parse_args()
    preprocess_dataset(args.data_root, args.out_root, device=args.device)
