import os
import random
from os.path import join as osp
from PIL import Image
import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from torchvision import transforms

class SF_XL_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_root='data/', img_size=256):
        self.file_list = os.listdir(osp(file_root, 't0'))
        
        self.pre_images = [osp(file_root, 't0', x) for x in self.file_list]
        self.post_images = [osp(file_root, 't1', x) for x in self.file_list]
        self.gts = [osp(file_root, 'mask', x) for x in self.file_list]
        # self.gts = [osp(file_root, 't0_mask', x.replace('.png', '_mask.png')) for x in self.file_list]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),  # Resize first
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),  # Resize masks
            transforms.ToTensor(),  # Convert to tensor
        ])
    
    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        # Load images
        pre_image = Image.open(self.pre_images[idx]).convert("RGB")
        post_image = Image.open(self.post_images[idx]).convert("RGB")
        label = Image.open(self.gts[idx]).convert("L")  # Open mask as grayscale
        
        # Apply transformations
        pre_image = self.transform(pre_image)
        post_image = self.transform(post_image)
        label = self.mask_transform(label)  # (1, H, W) mask, values 0 or 1 (normalized)

        return pre_image, post_image, label

    def apply_random_color_shift(self, img, mask, prob=0.3):
        """Applies random color shifts to change regions with a probability."""
        if random.random() > prob:
            return img  # 70% of the time, return unchanged image

        img = img.clone()  # Clone to avoid modifying original image

        # Generate random color shift values
        shift = torch.empty(3, 1, 1).uniform_(-0.3, 0.3)  # Random shift per channel

        # Apply shift only to changed areas (where mask is white)
        mask_binary = (mask > 0.5)  # Convert to binary mask
        img = img + (mask_binary * shift)  # Apply shift only on masked region
        img = torch.clamp(img, 0, 1)  # Ensure valid pixel range [0,1]

        return img

    def get_img_info(self, idx):
        img = Image.open(self.pre_images[idx])
        return {"height": img.height, "width": img.width}
