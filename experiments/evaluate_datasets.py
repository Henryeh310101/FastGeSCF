"""
test on standard SCD datasets and ChangeVPR (or own image pairs)
"""
import os 
import sys
from pathlib import Path
from os.path import join as osp
import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import random
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

from project_config import dataset_root, resolve_device
from framework import GeSCF
from py_utils.utils import calculate_metric

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def test_full_dataset(
    dataset,
    split=None,
    save_img=False,
    root=None,
    device="auto",
    sam2_checkpoint=None,
    outdoor_ckpt=None,
    indoor_ckpt=None,
    dry_run=False,
    max_items=None,
):
    if dataset == 'ChangeSim':
        case = 'Indoor'
    else:
        case = 'Outdoor'
    
    precisions = []
    recalls = []
    
    # example: test on VL-CMU-CD dataset
    if dataset == 'VL_CMU_CD':
        base_path = dataset_root(dataset, root)
        path_t0 = str(base_path / "test" / "t0")
        path_t1 = str(base_path / "test" / "t1")
        path_gt = str(base_path / "test" / "mask")
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
          
    elif dataset == 'PSCD':
        base_path = dataset_root(dataset, root)
        path_t0 = str(base_path / "t0")
        path_t1 = str(base_path / "t1")
        path_gt = str(base_path / "mask")
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
    elif dataset == 'St Lucia':
        split = split
        base_path = dataset_root(dataset, root)
        path_t0 = str(base_path / "t0")
        path_t1 = str(base_path / "t1")
        path_gt = str(base_path / "mask")
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
    elif dataset == 'Nordland':
        split = split
        base_path = dataset_root(dataset, root)
        path_t0 = str(base_path / "t0")
        path_t1 = str(base_path / "t1")
        path_gt = str(base_path / "mask")
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
    elif dataset == 'SF-XL':
        split = split
        base_path = dataset_root(dataset, root)
        path_t0 = str(base_path / "t0")
        path_t1 = str(base_path / "t1")
        path_gt = str(base_path / "mask")
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
    elif dataset == 'ChangeSim':
        split = split  # Keep this if it's used downstream
        base_path = str(dataset_root(dataset, root))
        warehouses = [6, 7, 8, 9]

        t0_images = []
        t1_images = []
        gt_images = []

        for wid in warehouses:
            path_t0 = osp(base_path, f"Warehouse_{wid}", "Seq_0_dark", "rgb")
            path_t1 = osp(base_path, f"Warehouse_{wid}", "Seq_0_dark", "t0", "rgb")
            path_gt = osp(base_path, f"Warehouse_{wid}", "Seq_0_dark", "change_segmentation")
            if not all(map(os.path.exists, [path_t0, path_t1, path_gt])):
                print(f"Skipping Warehouse_{wid} due to missing folders.")
                continue
            
            t0_list = sorted(os.listdir(path_t0))
            t1_list = sorted(os.listdir(path_t1))
            gt_list = sorted(os.listdir(path_gt))
            
            t0_images.extend([os.path.join(path_t0, f) for f in t0_list])
            t1_images.extend([os.path.join(path_t1, f) for f in t1_list])
            gt_images.extend([os.path.join(path_gt, f) for f in gt_list])

    if dry_run:
        logging.info(f"{dataset}: found {len(t0_images)} image pairs")
        return 0.0, 0.0, 0.0

    if max_items is not None:
        t0_images = t0_images[:max_items]
        t1_images = t1_images[:max_items]
        gt_images = gt_images[:max_items]

    model = GeSCF(
        dataset=dataset,
        feature_facet='key',
        case=case,
        device=device,
        sam2_checkpoint=sam2_checkpoint,
        outdoor_ckpt=outdoor_ckpt,
        indoor_ckpt=indoor_ckpt,
    )
    # vis_save_dir = "./original_results"
    # os.makedirs(vis_save_dir, exist_ok=True) 
    # total_points = 0
    time = 0
    pbar = tqdm(zip(t0_images, t1_images, gt_images), total=len(t0_images))
    for n, (t0, t1, gt) in enumerate(pbar):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dataset == 'ChangeSim':
            t0_path = t0
            t1_path = t1
            gt_path = gt
        else:
            t0_path = path_t0 + '/' + t0
            t1_path = path_t1 + '/' + t1
            gt_path = path_gt + '/' + gt
        
        gt = cv2.imread(gt_path, 0) / 255.   
        gt = cv2.resize(gt, (512,512))
        final_change_mask, p_time = model(t0_path, t1_path)
        time += p_time
        # final_change_mask = model(t0_path, t1_path)
        # total_points += points
        # save_name = os.path.splitext(t0)[0] + "_pred.png"
        # save_path = os.path.join(vis_save_dir, save_name)
        # cv2.imwrite(save_path, (final_change_mask.astype(np.uint8) * 255))
        
        prediction = final_change_mask
        precision, recall = calculate_metric(gt, prediction)
        
        precisions.append(precision)
        recalls.append(recall)
        # total_points += points
        
        current_precision = sum(precisions) / len(precisions)
        current_recall = sum(recalls) / len(recalls)
        current_f1score = 2 * (current_precision * current_recall) / (current_precision + current_recall)
        # current_avg_points = total_points / len(precisions)

        pbar.set_description(f"Processing {n+1}/{len(t0_images)}")
        pbar.set_postfix({
            "Precision": f"{current_precision:.4f}",
            "Recall": f"{current_recall:.4f}",
            "F1": f"{current_f1score:.4f}",
            # "Avg Points": f"{current_avg_points:.4f}",
        })
    
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    # iou = sum(ious) / len(ious)
    f1score = 2 * (precision * recall) / (precision + recall)
    del model
    return precision, recall, f1score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate FastGeSCF on SCD datasets")
    parser.add_argument("--datasets", nargs="+", default=['St Lucia', 'SF-XL', 'Nordland', 'VL_CMU_CD', 'PSCD', 'ChangeSim'])
    parser.add_argument("--data_root", type=str, default=None, help="Root for a single selected dataset")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, cuda:N, or cpu")
    parser.add_argument("--sam2_checkpoint", type=str, default=None, help="Path to SAM2.1 large checkpoint")
    parser.add_argument("--outdoor_ckpt", type=str, default=None, help="Path to outdoor FastGeSCF checkpoint")
    parser.add_argument("--indoor_ckpt", type=str, default=None, help="Path to indoor FastGeSCF checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Validate paths and image lists without loading models")
    parser.add_argument("--max_items", type=int, default=None, help="Limit evaluated image pairs")
    args = parser.parse_args()
    device = resolve_device(args.device)

    logging.info('FastGeSCF Testing')
    
    for dataset in args.datasets:
        precision, recall, f1score = test_full_dataset(
            dataset,
            root=args.data_root,
            device=device,
            sam2_checkpoint=args.sam2_checkpoint,
            outdoor_ckpt=args.outdoor_ckpt,
            indoor_ckpt=args.indoor_ckpt,
            dry_run=args.dry_run,
            max_items=args.max_items,
        )
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
