import os
import sys
from pathlib import Path
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import CHECKPOINT_DIR, resolve_device
from framework import GeSCF
from py_utils.vpr import load_model, get_descriptors
from datasets.video_dataset import VideoVPRDataset

warnings.filterwarnings("ignore")

def validate_frame_dir(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        raise FileNotFoundError(f"{label} has no .jpg or .png frames: {path}")
    return len(files)

def filter_path(path):
    filtered_path = {}
    for q_idx, r_idx in path:
        filtered_path[q_idx] = r_idx
    return sorted(filtered_path.items())

def get_dtw_matching(vpr_model, query_dir, ref_dir, device):
    dataset = VideoVPRDataset(ref_dir=ref_dir, query_dir=query_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    descriptors = get_descriptors(vpr_model, loader, device)
    n_query, n_ref = dataset.num_queries, dataset.num_references
    
    r_list = descriptors[:n_ref]
    q_list = descriptors[n_ref:]

    _, path = fastdtw(q_list, r_list)
    path = filter_path(path)
    return path


def detect_changes_from_matched_pairs(model, match_path, query_dir, ref_dir, output_dir, resize=(512, 512), fps=30):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "change_detection_result.mp4")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    query_files = sorted([f for f in os.listdir(query_dir) if f.lower().endswith(('.jpg', '.png'))])
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.jpg', '.png'))])

    width, height = resize
    frame_size = (width * 2, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for i, (q_idx, r_idx) in enumerate(tqdm(match_path, desc="Writing Video")):
        # q_img_path = os.path.join(query_dir, f"{q_idx:05d}.png")
        # r_img_path = os.path.join(ref_dir, f"{r_idx:05d}.png")
        q_img_path = os.path.join(query_dir, query_files[q_idx])
        r_img_path = os.path.join(ref_dir, ref_files[r_idx])
        if not os.path.exists(q_img_path) or not os.path.exists(r_img_path):
            continue

        # Run GeSCF model
        final_change_mask = model(q_img_path, r_img_path)
        prediction = final_change_mask[0]

        # Load original images
        query_img = cv2.imread(q_img_path)
        ref_img = cv2.imread(r_img_path)
        query_img = cv2.resize(query_img, resize)
        ref_img = cv2.resize(ref_img, resize)

        # Create overlay
        pred_mask = prediction.astype(np.uint8) * 255
        pred_mask_colored = np.zeros_like(ref_img)
        pred_mask_colored[:, :, 2] = pred_mask  # Red channel

        overlay = cv2.addWeighted(ref_img, 0.7, pred_mask_colored, 0.3, 0)

        # Concatenate query and overlay side-by-side
        combined = np.concatenate((query_img, overlay), axis=1)
        video_writer.write(combined)

        overlay_path = os.path.join(overlay_dir, f"{i:05d}.jpg")
        cv2.imwrite(overlay_path, overlay)

    video_writer.release()
    print(f"\nVideo written to: {video_path}")
    print(f"Overlays saved to: {overlay_dir}")

def main(args):
    args.device = resolve_device(args.device)
    if args.dry_run:
        query_count = validate_frame_dir(args.query_dir, "query frame directory")
        ref_count = validate_frame_dir(args.ref_dir, "reference frame directory")
        if not os.path.isfile(args.vpr_ckpt_path):
            raise FileNotFoundError(f"VPR checkpoint not found: {args.vpr_ckpt_path}")
        print(f"Query frames: {query_count}")
        print(f"Reference frames: {ref_count}")
        print("Dry run complete.")
        return

    print("=== Step 1: Loading VPR model ===")
    vpr_model = load_model(args.vpr_ckpt_path, args.device).to(args.device)

    print("=== Step 2: Matching with FastDTW ===")
    match_path = get_dtw_matching(vpr_model, args.query_dir, args.ref_dir, args.device)

    print("=== Step 3: Running GeSCF and Creating Video ===")
    cd_model = GeSCF(
        dataset=args.dataset,
        case=args.case,
        device=args.device,
        sam2_checkpoint=args.sam2_checkpoint,
        outdoor_ckpt=args.outdoor_ckpt,
        indoor_ckpt=args.indoor_ckpt,
    )
    detect_changes_from_matched_pairs(cd_model, match_path, args.query_dir, args.ref_dir, args.output_dir)

    print(f"\nDone. Side-by-side result video saved to: {args.output_dir}/change_detection_result.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VPR + Change Detection on Two Image Directories")
    parser.add_argument('--query_dir', type=str, required=True, help="Directory with query images)")
    parser.add_argument('--ref_dir', type=str, required=True, help="Directory with reference images")
    parser.add_argument('--vpr_ckpt_path', type=str, default=str(CHECKPOINT_DIR / "dino_salad.ckpt"), help="Path to the VPR model checkpoint")
    parser.add_argument('--device', type=str, default='auto', help="Device: auto, cuda, cuda:N, or cpu")
    parser.add_argument('--dataset', type=str, default='SF-XL', help="Dataset name used for registration/refinement rules")
    parser.add_argument('--case', choices=['Outdoor', 'Indoor'], default='Outdoor', help="FastGeSCF checkpoint family")
    parser.add_argument('--sam2_checkpoint', type=str, default=None, help="Path to SAM2.1 large checkpoint")
    parser.add_argument('--outdoor_ckpt', type=str, default=None, help="Path to outdoor FastGeSCF checkpoint")
    parser.add_argument('--indoor_ckpt', type=str, default=None, help="Path to indoor FastGeSCF checkpoint")
    parser.add_argument('--output_dir', type=str, default="./change_results", help="Output dir for results")
    parser.add_argument('--dry_run', action='store_true', help="Validate inputs without loading models")

    args = parser.parse_args()
    main(args)
