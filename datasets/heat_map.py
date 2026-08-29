import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def read_centroids(txt_path):
    """Read centroid coordinates from a .txt file."""
    centroids = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(int, line.split(','))
                centroids.append((x, y))
    return centroids

def generate_centroid_heatmap(shape, centroids, radius=5, gaussian=True):
    """
    Create a heatmap from a list of (x, y) centroid points.

    Args:
        shape (tuple): (height, width) of the heatmap
        centroids (list of tuples): [(x1, y1), (x2, y2), ...]
        radius (int): Radius of the blobs
        gaussian (bool): Use Gaussian blur or solid circle

    Returns:
        np.ndarray: Heatmap with peaks at centroid locations
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    for x, y in centroids:
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            if gaussian:
                temp = np.zeros_like(heatmap)
                cv2.circle(temp, (x, y), radius, 1, -1)
                heatmap += cv2.GaussianBlur(temp, (0, 0), sigmaX=radius / 2)
            else:
                cv2.circle(heatmap, (x, y), radius, 1, -1)

    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a centroid heatmap visualization")
    parser.add_argument("--txt_file", required=True, help="Centroid .txt file")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--output", default="centroid_heatmap_visualization.png")
    args = parser.parse_args()

    centroids = read_centroids(args.txt_file)
    heatmap = generate_centroid_heatmap((args.height, args.width), centroids, radius=args.radius, gaussian=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(f"Heatmap with {len(centroids)} centroids")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
