import os
import cv2
import argparse
from tqdm import tqdm

def generate_video_from_path(filtered_path, q_dir, r_dir, save_path, resize=(512, 512), fps=10):
    frames = []
    
    def numeric_sort(file_list):
            return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    
    q_files = numeric_sort(os.listdir(q_dir))
    r_files = numeric_sort(os.listdir(r_dir))
    
    for q_idx, r_idx in tqdm(filtered_path, desc="Generating video"):
        q_img_path = os.path.join(q_dir, q_files[q_idx])
        r_img_path = os.path.join(r_dir, r_files[r_idx])

        q_img = cv2.imread(q_img_path)
        r_img = cv2.imread(r_img_path)

        if q_img is None or r_img is None:
            print(f"Skipping: {q_img_path}, {r_img_path}")
            continue

        q_img = cv2.resize(q_img, resize)
        r_img = cv2.resize(r_img, resize)

        combined = cv2.hconcat([q_img, r_img])
        frames.append(combined)

    # Define video writer
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Video saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from aligned query-ref pairs")
    parser.add_argument('--q_path', type=str, required=True, help="Path to query image directory")
    parser.add_argument('--r_path', type=str, required=True, help="Path to reference image directory")
    parser.add_argument('--output', type=str, default="warehouse6.mp4", help="Output video path")
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help="Resize each image (width height)")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second in output video")
    parser.add_argument('--path_file', type=str, required=True, help="Path to .txt or .npy file containing filtered_path")

    args = parser.parse_args()

    # Load path (can be .npy or .txt with lines of `q_idx r_idx`)
    if args.path_file.endswith(".npy"):
        import numpy as np
        filtered_path = np.load(args.path_file, allow_pickle=True)
    else:
        filtered_path = []
        with open(args.path_file, 'r') as f:
            for line in f:
                q_idx, r_idx = map(int, line.strip().split())
                filtered_path.append((q_idx, r_idx))

    generate_video_from_path(
        filtered_path=filtered_path,
        q_dir=args.q_path,
        r_dir=args.r_path,
        save_path=args.output,
        resize=tuple(args.resize),
        fps=args.fps
    )