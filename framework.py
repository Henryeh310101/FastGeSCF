"""
Generalizable Scene Change Detection Framework (GeSCF)
"""
import cv2
import numpy as np
import logging
import time
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from scipy.stats import skew
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch
import torchvision
import torch.nn as nn

from project_config import CHECKPOINT_DIR, RESULTS_DIR, require_file, resolve_device
from py_utils.utils import calculate_iou

## modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from model.CD_model import CrossAttention
from registration import coarse_transform

def downsample_points(input_points, input_labels, max_points=50):
    if len(input_points) <= max_points:
        return input_points, input_labels

    # Flatten from (N, 1, 2) to (N, 2)
    points = input_points.reshape(-1, 2)

    # KMeans clustering
    n_clusters = len(input_points) // 2
    n_clusters = min(n_clusters, 200)
    
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(points)

    # For each cluster center, find the nearest original point
    distances = cdist(kmeans.cluster_centers_, points)
    nearest_indices = np.argmin(distances, axis=1)
    selected_points = points[nearest_indices].reshape(-1, 1, 2)

    selected_labels = input_labels[nearest_indices]

    return selected_points, selected_labels

class GeSCF(nn.Module):
    def __init__(
        self,
        dataset='SF-XL',
        feature_facet='key',
        feature_layer=24,
        embedding_layer=4,
        case='Outdoor',
        device=None,
        sam2_checkpoint=None,
        outdoor_ckpt=None,
        indoor_ckpt=None,
    ):
        # assert dataset in ['VL_CMU_CD', 'TSUNAMI', 'ChangeSim', 'ChangeVPR', 'Remote_Sensing', 'Random', 'PSCD']
        assert feature_facet in ['query', 'key', 'value']
        # assert feature_layer in [i for i in range(1,33)] # ViT-Huge has 32 layers
        # assert embedding_layer in [i for i in range(1,33)] # ViT-Huge has 32 layers
        super(GeSCF, self).__init__()
        
        self.dataset = dataset
        self.device = resolve_device(device)
        self.dataset_bias = True if self.dataset == 'VL_CMU_CD' else False
        logging.info(f'dataset name: {dataset}')
        
        self.img_size = (512,512) if self.dataset != 'TSUNAMI' else (224, 224)
        
        # default settings
        self.feature_facet = feature_facet
        self.feature_layer = feature_layer
        self.embedding_layer = embedding_layer
        self.alpha_t = 0.5
        
        # build SAM
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = sam2_checkpoint or CHECKPOINT_DIR / "sam2.1_hiera_large.pt"
        sam2_checkpoint = require_file(sam2_checkpoint, "SAM2 checkpoint")
        self.sam_backbone = build_sam2(
            sam2_model_cfg,
            str(sam2_checkpoint),
            device=self.device,
            apply_postprocessing=False,
        )
        
        # build automatic mask generator
        self.image_mask_generator = SAM2ImagePredictor(self.sam_backbone, max_hole_area=0, max_sprinkle_area=50)
        # self.image_mask_generator = SAM2ImagePredictor(self.sam_backbone, max_hole_area=100, max_sprinkle_area=100)
        
        # build pseudo_generator
        self.robust_generator = CrossAttention(
            target_shp=self.img_size,
            device=self.device,
            sam2_checkpoint=sam2_checkpoint,
        )
        
        if case == 'Outdoor':
            robust_ckpt = outdoor_ckpt or RESULTS_DIR / "RobustViT_vl-cmu-cd" / "best_model.pth"
        else:
            robust_ckpt = indoor_ckpt or RESULTS_DIR / "RobustViT_changesim" / "best_model.pth"
        robust_ckpt = require_file(robust_ckpt, "FastGeSCF robust-generator checkpoint")
        self.robust_generator.load_state_dict(torch.load(robust_ckpt, map_location="cpu"))
    
    def load_img(self, img_path):
        # load rgb/grayscale image of the given image path
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, self.img_size) / 255.
        
        rgb_img = np.array(rgb_img)
        gray_img = np.array(gray_img)
        input = self.transform()(rgb_img).unsqueeze(0)
        
        return rgb_img, gray_img, input
        
    
    def transform(self):
        tr_lst = [torchvision.transforms.ToTensor()]
        tr_lst.append(torchvision.transforms.Resize(self.img_size))
        tr_lst.append(torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],))
        tr = torchvision.transforms.Compose(tr_lst)
        return tr
        
        
    def get_robust_mask(self, sim_map1, sim_map2):
        sim_map1 = sim_map1.detach().cpu().numpy()[0]
        sim_map2 = sim_map2.detach().cpu().numpy()[0]
        
        # Get moderate mask
        mask1 = np.argmax(sim_map1, axis=-1).astype(np.uint8)
        mask2 = np.argmax(sim_map2, axis=-1).astype(np.uint8)
        mask = np.logical_or(mask1, mask2).astype(np.uint8)
        
        return mask, mask1, mask2
        

    def generate_prompts_from_masks(self, mask, grid_size=32):
        H, W = mask.shape
        patch_h = H // grid_size
        patch_w = W // grid_size

        input_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                patch = mask[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                if np.any(patch):
                    # Center of the patch
                    y = i * patch_h + patch_h // 2
                    x = j * patch_w + patch_w // 2
                    input_points.append([x, y])

        input_points = np.array(input_points).reshape(-1, 1, 2)
        input_labels = np.ones((input_points.shape[0], 1), dtype=np.int32)

        return input_points, input_labels

    def forward(self, img_t0_path, img_t1_path):
        '''generate final change mask of given image pairs'''
        img_t0, gray_img_t0, input_t0 = self.load_img(img_t0_path)
        img_t1, gray_img_t1, input_t1 = self.load_img(img_t1_path)
        
        aligned_img_t1, H, flag = coarse_transform(
            self.dataset,
            self.img_size,
            img_t0,
            img_t1,
            gray_img_t0,
            gray_img_t1,
            device=self.device,
        )
        if flag or self.dataset == 'ChangeSim':
            img_t1 = np.array(aligned_img_t1)
            input_t1 = self.transform()(img_t1).unsqueeze(0)
        # flag = False
        # aligned_img_t1 = img_t1
        ##################################################
        # Initial Pseudo-Mask Generation 
        ##################################################
        self.robust_generator.to(self.device)
        sim_map1, sim_map2, _, _ = self.robust_generator(input_t0.to(self.device), input_t1.to(self.device))
        mask, mask1, mask2 = self.get_robust_mask(sim_map1, sim_map2)

        sim_map = sim_map1.detach().cpu().numpy()[0]
        ##################################################
        # Generate Input Point for SAM
        ##################################################
        start = time.time()
        self.image_mask_generator.set_image(img_t0)
        input_points1, input_labels1 = self.generate_prompts_from_masks(mask1)
        input_points2, input_labels2 = self.generate_prompts_from_masks(mask2)
        
        input_points1, input_labels1 = downsample_points(input_points1, input_labels1)
        input_points2, input_labels2 = downsample_points(input_points2, input_labels2)

        ### Masks_t0 ###
        if len(input_points1) != 0:
            masks_t0, _, _ = self.image_mask_generator.predict(
                point_coords=input_points1,
                point_labels=input_labels1,
                multimask_output=False,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            masks_t0 = np.expand_dims(np.zeros_like(mask1, dtype=np.uint8), axis=0)
        
        ### Masks_t1 ###
        if len(input_points2) != 0:
            self.image_mask_generator.set_image(img_t1)
            masks_t1, _, _ = self.image_mask_generator.predict(
                point_coords=input_points2,
                point_labels=input_labels2,
                multimask_output=False,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            masks_t1 = np.expand_dims(np.zeros_like(mask1, dtype=np.uint8), axis=0)
        if len(input_points1) <= 1:
            masks_t0 = np.expand_dims(masks_t0, axis=0)
        if len(input_points2) <= 1:
            masks_t1 = np.expand_dims(masks_t1, axis=0)
        ##################################################
        # Refine Noises and out-of-view (oov) regions
        ##################################################
        binary_mask_outliers = np.zeros_like(sim_map, dtype=np.uint8)
        
        if self.dataset == 'VL_CMU_CD':
            oov_mask = np.all(img_t0 == [0, 0, 0], axis=-1)
            binary_mask_outliers[oov_mask] = 0
            mask[oov_mask] = 0
        if flag:
            warped_oov_mask = np.all(img_t1 == [0, 0, 0], axis=-1)
            binary_mask_outliers[warped_oov_mask] = 0
            mask[warped_oov_mask] = 0

            # refine initial pseudo mask 1
            padding_size = 10
            kernel_size = 2 * padding_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            warped_oov_mask_uint8 = warped_oov_mask.astype(np.uint8) * 255
            dilated_mask = cv2.dilate(warped_oov_mask_uint8, kernel, iterations=1)
            dilated_mask = dilated_mask.astype(bool)
            binary_mask_outliers[dilated_mask] = 0
            mask[dilated_mask] = 0
                
        initial_pseudo_mask = mask
        
        # refine initial pseudo mask 2
        initial_pseudo_mask = initial_pseudo_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        initial_pseudo_mask = cv2.morphologyEx(initial_pseudo_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(initial_pseudo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # threshold for minimum area to keep
                cv2.drawContours(initial_pseudo_mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)


        ####################################################
        # Geometric Mask Matching 
        ####################################################

        mask_idx_t0 = []
        mask_idx_t1 = []
        
        # geometric intersection matching (t0 > t1)
        for i in range(len(masks_t0)):
            iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t0[i].squeeze(0))
            if iou >= self.alpha_t: 
                mask_idx_t0.append(i)
                    
        x = np.zeros_like(initial_pseudo_mask)
        for j in mask_idx_t0:
            x = np.logical_or(x, masks_t0[j].squeeze(0))
   
        # geometric intersection matching (t1 > t0)
        if not self.dataset_bias:
            for k in range(len(masks_t1)):
                iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t1[k].squeeze(0))
                if iou >= self.alpha_t:
                    mask_idx_t1.append(k)

            y = np.zeros_like(initial_pseudo_mask)
            
            for l in mask_idx_t1:
                y = np.logical_or(y, masks_t1[l].squeeze(0))
            final_change_mask = np.logical_or(x, y)
            
        else:
            final_change_mask = x

        return final_change_mask, time.time()-start
