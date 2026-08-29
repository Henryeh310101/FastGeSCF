import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # Assuming you're using SAM2 from Meta’s structure

np.random.seed(3)

device = "cuda:3"

def show_mask(mask, ax, random_color=True, borders = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # if borders:
    #     import cv2
    #     contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #     # Try to smooth contours
    #     contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    #     mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_path="prompt_mask.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Overlay all masks
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), borders=borders)

    # Show points if provided
    # if point_coords is not None and input_labels is not None:
    #     show_points(point_coords, input_labels, plt.gca())

    # Show box if provided
    # if box_coords is not None:
    #     show_box(box_coords, plt.gca())

    plt.axis('off')
    # plt.title("All Masks", fontsize=18)
    plt.savefig(save_path)
    plt.close()


# Load image
image = Image.open("/media/tesla/E/henry/data/Nordland/t0/00000137.png")
image = np.array(image.convert("RGB"))

# Build SAM2 model
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cpu", apply_postprocessing=False)

# Wrap model in predictor
predictor = SAM2ImagePredictor(sam2, max_hole_area=0, max_sprinkle_area=50)
predictor.set_image(image)

# Define 32x32 grid points
height, width, _ = image.shape
grid_rows, grid_cols = 3, 3
y_coords = np.linspace(0, height - 1, grid_rows, dtype=int)
x_coords = np.linspace(0, width - 1, grid_cols, dtype=int)
grid_points = np.array([[x, y] for y in y_coords for x in x_coords])

# Convert to SAM input format
input_points = grid_points[:, None, :]  # shape (N, 1, 2)
input_labels = np.ones((len(grid_points), 1), dtype=np.int32)  # All foreground

# Run prediction
start = time.time()
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
)
end = time.time()
print("Time generated mask: ", end - start)

# Format for visualization
anns = [{"segmentation": masks[i], "area": masks[i].sum()} for i in range(masks.shape[0])]

# Plot result
show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels, borders=False)
