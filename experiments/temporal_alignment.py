import os
import sys
from pathlib import Path
import cv2
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from fastdtw import fastdtw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import CHECKPOINT_DIR, resolve_device
from py_utils.vpr import load_model, get_descriptors, get_nearest
from py_utils.utils import calculate_metric
from datasets.vpr_dataset import VPR_Dataset
from datasets.cd_dataset import CD_Dataset

import warnings
warnings.filterwarnings("ignore")

def validate_warehouse_layout(args, warehouse_id):
    env = args.env
    required_paths = [
        os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/rgb"),
        os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0/rgb"),
        os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/change_segmentation"),
        os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0/trajectory.txt"),
        os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/trajectory.txt"),
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Warehouse {warehouse_id} is missing required paths: {missing}")

def load_trajectory(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0])
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            data[idx] = pos
    return data

def compute_recalls_from_dtw_path(path, query_traj, ref_traj, threshold=1.0):
    recall_counts = np.zeros(3)  # Recall@1, 5, 10, 20
    top_ks = [1, 5, 10]

    for q_idx, matched_r_idx in path:
        q_pos = np.array(query_traj[q_idx][:3])
        
        for i, k in enumerate(top_ks):
            # Create a window around the matched reference index
            half_k = k // 2
            min_idx = max(0, matched_r_idx - half_k)
            max_idx = min(len(ref_traj), matched_r_idx + half_k + 1)
            window_ref_positions = [ref_traj[i][:3] for i in range(min_idx, max_idx)]

            # Check if any distance in this window is within threshold
            found = any(
                np.linalg.norm(q_pos - np.array(ref_pos)) < threshold
                for ref_pos in window_ref_positions
            )
            if found:
                recall_counts[i] += 1

    recalls = recall_counts / len(path)
    return recalls  # [recall@1, recall@5, recall@10, recall@20]


def compute_recalls(indices, query_traj, ref_traj, threshold=1.0):
    recalls = np.zeros(3)
    num_queries = indices.shape[0]

    for q_idx in range(num_queries):
        q_pos = query_traj[q_idx][:3]
        found = [False] * 3
        for i, k in enumerate([1, 5, 10]):
            for r_idx in indices[q_idx, :k]:
                r_pos = ref_traj[r_idx][:3]
                dist = np.linalg.norm(q_pos - r_pos)
                if dist < threshold:
                    found[i] = True
                    break
        recalls += found

    return recalls / num_queries  # normalize to get recall rate

def filter_path(path):
    filtered_path = {}
    for q_idx, r_idx in path:
        filtered_path[q_idx] = r_idx
    return sorted(filtered_path.items())

def evaluate_on_warehouse(args, warehouse_id):
    print(f"\n--- Evaluating Warehouse {warehouse_id} ---")
    env = args.env
    r_path = os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/rgb")
    q_path = os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0/rgb")
    m_path = os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/change_segmentation")
    
    # model = GeSCF(dataset='ChangeSim', case='Indoor')

    vpr_dataset = VPR_Dataset(ref_dir=r_path, query_dir=q_path)
    vpr_data_loader = DataLoader(vpr_dataset, num_workers=16, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    n_query, n_ref = vpr_dataset.num_queries, vpr_dataset.num_references
    vpr_model = load_model(args.ckpt_path, device=args.device).to(args.device)
    descriptors = get_descriptors(vpr_model, vpr_data_loader, args.device)
    
    r_list = descriptors[:n_ref]
    q_list = descriptors[n_ref:]
    
    _, dtw_path = fastdtw(q_list, r_list)
    path = get_nearest(q_list, r_list, k=10)
    
    filtered_dtw_path = filter_path(dtw_path)
    # filtered_path = filter_path(path)
        # Save matching path to a .txt file
    dtw_save_path = os.path.join(args.output_dir, f"{env}_results", "vpr_fastdtw", f"warehouse_{warehouse_id}_path_plt.txt")
    save_path = os.path.join(args.output_dir, f"{env}_results", "vpr", f"warehouse_{warehouse_id}_path_plt.txt")
    
    os.makedirs(os.path.dirname(dtw_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(dtw_save_path, 'w') as f:
        f.write("# Format: query_index -> reference_index\n")
        for q_idx, r_idx in filtered_dtw_path:
            f.write(f"{q_idx} -> {r_idx}\n")
            
    # with open(save_path, 'w') as f:
    #     f.write("# Format: query_index -> reference_index\n")
    #     for q_idx, r_idx in path:
    #         f.write(f"{q_idx} -> {r_idx}\n")

    # print(f"Saved DTW path to: {save_path}")

    query_traj_path = os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0/trajectory.txt")
    ref_traj_path = os.path.join(args.base, f"Warehouse_{warehouse_id}/Seq_0_{env}/trajectory.txt")
    query_traj = load_trajectory(query_traj_path)
    ref_traj = load_trajectory(ref_traj_path)
    
    # Compute recall
    recalls_dtw = compute_recalls_from_dtw_path(filtered_dtw_path, query_traj, ref_traj)
    recalls = compute_recalls(path, query_traj, ref_traj)
    
    return recalls_dtw, recalls

def main(args):
    args.device = resolve_device(args.device)
    if args.dry_run:
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"VPR checkpoint not found: {args.ckpt_path}")
        for wid in args.warehouse_ids:
            validate_warehouse_layout(args, wid)
        print(f"Validated warehouses: {args.warehouse_ids}")
        print("Dry run complete.")
        return

    all_recalls_dtw = []
    all_recalls_vpr = []

    for wid in args.warehouse_ids:
        recalls_dtw, recalls_vpr = evaluate_on_warehouse(args, wid)
        all_recalls_dtw.append(recalls_dtw)
        all_recalls_vpr.append(recalls_vpr)

        print(f"\nWarehouse {wid} ({args.env})")
        print(f"  DTW  Recall@1/5/10: {recalls_dtw}")
        print(f"  VPR  Recall@1/5/10: {recalls_vpr}")

    mean_dtw = np.mean(np.stack(all_recalls_dtw, axis=0), axis=0) if all_recalls_dtw else np.zeros(3)
    mean_vpr = np.mean(np.stack(all_recalls_vpr, axis=0), axis=0) if all_recalls_vpr else np.zeros(3)

    print("\n=== Final Mean Recalls ===")
    print(f"DTW Recall@1/5/10: {mean_dtw}")
    print(f"VPR Recall@1/5/10: {mean_vpr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VPR + CD across multiple warehouses")
    parser.add_argument('--base', type=str, required=True, help="Base path for ChangeSim query images and masks")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--env', type=str, default='dust', help="Query Environment, can be dark or dust")
    parser.add_argument('--device', type=str, default='auto', help="Device to run the model: auto, cuda, cuda:N, or cpu")
    parser.add_argument('--ckpt_path', type=str, default=str(CHECKPOINT_DIR / "dino_salad.ckpt"), help="Path to the VPR model checkpoint")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Directory for alignment result files")
    parser.add_argument('--warehouse_ids', type=int, nargs='+', default=[6,7,8,9],
                        help="Which warehouses to evaluate, e.g. --warehouse_ids 8 9")
    parser.add_argument('--dry_run', action='store_true', help="Validate inputs without loading models")

    args = parser.parse_args()
    main(args)
