import os
import random
import numpy as np
from os.path import join as osp
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as tvff
import albumentations as A
from albumentations.pytorch import ToTensorV2

from project_config import dataset_root

class ChangeSimDataset(torch.utils.data.Dataset):
    def __init__(self, file_root=None, img_size=512, mode='train'):
        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"
        if file_root is None:
            file_root = dataset_root("ChangeSim" if mode == "test" else "changesim")
        self.mode = mode
        self.img_size = img_size
        self.hflip_prob = 0.5
        
        self.pre_images = []
        self.post_images = []
        self.gts = []
        ### Testing Set ###
        if self.mode == 'test':
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            for house_idx in range(6,10):
                base_path = osp(file_root, f'Warehouse_{house_idx}', 'Seq_0')
                t0_path = osp(base_path, 'rgb')
                t1_path = osp(base_path, 't0', 'rgb')
                mask_path = osp(base_path, 'change_segmentation')

                if not all(map(os.path.exists, [t0_path, t1_path, mask_path])):
                    continue
                filenames = sorted(os.listdir(t0_path))
                for fname in filenames:
                    self.pre_images.append(osp(t0_path, fname))
                    self.post_images.append(osp(t1_path, fname))
                    self.gts.append(osp(mask_path, fname))
        ### Training Set ###
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],additional_targets={'image0': 'image'})
            for house_idx in range(6):  # 0 to 5 inclusive
                base_path = osp(file_root, f'Warehouse_{house_idx}', 'Seq_0')
                t0_path = osp(base_path, 'rgb')
                t1_path = osp(base_path, 't0', 'rgb_aligned')
                mask_path = osp(base_path, 'change_segmentation')

                if not all(map(os.path.exists, [t0_path, t1_path, mask_path])):
                    continue

                filenames = sorted(os.listdir(t0_path))
                n_total = len(filenames)
                n_val = max(1, int(0.1 * n_total))  # at least 1 image

                if self.mode == 'train':
                    selected = filenames[:-n_val]
                elif self.mode == 'val':
                    selected = filenames[-n_val:]

                for fname in selected:
                    self.pre_images.append(osp(t0_path, fname))
                    self.post_images.append(osp(t1_path, fname))
                    self.gts.append(osp(mask_path, fname))

        # self.transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size), antialias=True),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

        # self.mask_transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size), antialias=True),
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        t0_image = np.array(Image.open(self.pre_images[idx]).convert("RGB"))
        t1_image = np.array(Image.open(self.post_images[idx]).convert("RGB"))
        mask_image = np.array(Image.open(self.gts[idx]).convert("L"))
        mask_image = (mask_image != 0).astype(np.uint8)  # Now values are 0 or 1

        # Stack the images together to apply identical augmentation
        combined = {
            'image': t0_image,
            'image0': t1_image,
            'mask': mask_image
        }

        augmented = self.aug(**combined)

        t0_image = augmented['image']
        t1_image = augmented['image0']
        mask_image = augmented['mask'].unsqueeze(0).float()  # Keep it (1, H, W)

        return t0_image, t1_image, mask_image

    def get_img_info(self, idx):
        img = Image.open(self.pre_images[idx])
        return {"height": img.height, "width": img.width}

# Training set
# train_set = ChangeSimDataset(mode='train')

# # Validation set
# val_set = ChangeSimDataset(mode='val')

# test_set = ChangeSimDataset(mode='test', file_root="/path/to/Query_Seq_Test")
# print(len(train_set))
# print(len(val_set))
# print(len(test_set))
