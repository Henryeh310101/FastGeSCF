"""
test on standard SCD datasets and ChangeVPR (or own image pairs)
"""
import os 
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework import GeSCF
from py_utils.utils import calculate_metric, show_mask_new

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def validate_image(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    if cv2.imread(path) is None:
        raise ValueError(f"{label} is not a readable image: {path}")

def test_single_image(
    img_t0_path,
    img_t1_path,
    gt_path=None,
    output_path="comparison.png",
    img_size=(512,512),
    dataset="SF-XL",
    case="Outdoor",
    device="auto",
    sam2_checkpoint=None,
    outdoor_ckpt=None,
    indoor_ckpt=None,
):
    model = GeSCF(
        dataset=dataset,
        case=case,
        device=device,
        sam2_checkpoint=sam2_checkpoint,
        outdoor_ckpt=outdoor_ckpt,
        indoor_ckpt=indoor_ckpt,
    )
    
    # load image pairs
    img_t0 = cv2.imread(img_t0_path)
    img_t0 = cv2.resize(img_t0, img_size)
    rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t1 = cv2.imread(img_t1_path)
    img_t1 = cv2.resize(img_t1, img_size)
    rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

    # inference
    final_change_mask = model(img_t0_path, img_t1_path)
    final_change_mask = final_change_mask[0]
    if gt_path:
        gt = cv2.imread(gt_path, 0) / 255.
        gt = cv2.resize(gt, img_size)
        gt[gt > 0] = 1.0
        gt_vis = (gt * 255).astype(np.uint8)        
        # cv2.imwrite("mask.png", gt_vis)
        precision, recall = calculate_metric(gt, final_change_mask)
        f1score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    # visualization
    fig = plt.figure(figsize=(12,4))
    fig.add_subplot(141)
    plt.title('img t0')
    plt.imshow(rgb_img_t0)
    plt.axis('off')

    fig.add_subplot(142)
    plt.title('img t1')
    plt.imshow(rgb_img_t1)
    plt.axis('off')
    
    ax3 = fig.add_subplot(143)
    plt.title('FastGeSCF')
    plt.imshow(rgb_img_t0)
    show_mask_new(final_change_mask.astype(np.float32), plt.gca())
    plt.axis('off')
    
    if gt_path:
        fig.add_subplot(144)
        plt.title('GT')
        plt.imshow(rgb_img_t0)
        show_mask_new(gt.astype(np.float32), plt.gca())
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path)
    
    del model
    if gt_path:
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
        return precision, recall, f1score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FastGeSCF on one image pair")
    parser.add_argument("--t0", required=True, help="Path to the reference/time-0 image")
    parser.add_argument("--t1", required=True, help="Path to the query/time-1 image")
    parser.add_argument("--gt", default=None, help="Optional ground-truth mask path")
    parser.add_argument("--dataset", default="SF-XL", help="Dataset name used for registration/refinement rules")
    parser.add_argument("--case", choices=["Outdoor", "Indoor"], default="Outdoor", help="Checkpoint family")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cuda:N, or cpu")
    parser.add_argument("--sam2_checkpoint", default=None, help="Path to SAM2.1 large checkpoint")
    parser.add_argument("--outdoor_ckpt", default=None, help="Path to outdoor FastGeSCF checkpoint")
    parser.add_argument("--indoor_ckpt", default=None, help="Path to indoor FastGeSCF checkpoint")
    parser.add_argument("--output", default="comparison.png", help="Output visualization path")
    parser.add_argument("--dry_run", action="store_true", help="Validate input image paths without loading models")
    args = parser.parse_args()

    if args.dry_run:
        validate_image(args.t0, "t0 image")
        validate_image(args.t1, "t1 image")
        if args.gt:
            validate_image(args.gt, "ground-truth mask")
        print("Dry run complete.")
        raise SystemExit(0)

    test_single_image(
        args.t0,
        args.t1,
        args.gt,
        output_path=args.output,
        dataset=args.dataset,
        case=args.case,
        device=args.device,
        sam2_checkpoint=args.sam2_checkpoint,
        outdoor_ckpt=args.outdoor_ckpt,
        indoor_ckpt=args.indoor_ckpt,
    )
