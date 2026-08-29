import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class CD_Dataset(Dataset):
    def __init__(self, query_dir, ref_dir, mask_dir, path=None, transform=None):
        """
        Dataset for change detection using aligned query and reference frames.
        :param query_dir: Path to the directory containing query frames.
        :param ref_dir: Path to the directory containing reference frames.
        :param path: List of tuples representing the aligned indices (from fastdtw).
        :param transform: Transform to apply to the images.
        """
        self.query_dir = query_dir
        self.ref_dir = ref_dir
        self.mask_dir = mask_dir
        self.path = path  # Aligned indices from fastdtw
        self.transform = transform
        
        def numeric_sort(file_list):
            return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
        
        self.query_files = numeric_sort(os.listdir(query_dir))
        self.ref_files = numeric_sort(os.listdir(ref_dir))
        self.mask_files = numeric_sort(os.listdir(mask_dir))

    def __len__(self):  
        return len(self.query_files)

    def __getitem__(self, idx):
        q_idx = self.path[idx][0]
        ref_idx = self.path[idx][1]
        query_path = os.path.join(self.query_dir, self.query_files[q_idx])
        ref_path = os.path.join(self.ref_dir, self.ref_files[ref_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[q_idx])

        return query_path, ref_path, mask_path

    def get_queries(self, selected_idx, img_size=None):
        q_images_paths = [self.query_files[i] for i in selected_idx]
        q_images = []

        for img_path in q_images_paths:
            img_path = self.query_dir + img_path
            with Image.open(img_path) as img:  # Automatically closes after exiting 'with' block
                if img_size:
                    img = img.resize(img_size, Image.ANTIALIAS)
                q_images.append(img.copy())  # Copy image to avoid issues after 'with' block

        return q_images

    def get_references(self, selected_idx, img_size=None):
        # Directly select images based on indices
        ref_images_paths = [self.ref_files[i] for i in selected_idx]
        ref_images = []

        for img_path in ref_images_paths:
            img_path = self.ref_dir + img_path
            with Image.open(img_path) as img:  # Automatically closes after exiting 'with' block
                if img_size:
                    img = img.resize(img_size, Image.ANTIALIAS)
                ref_images.append(img.copy())
                
        return ref_images