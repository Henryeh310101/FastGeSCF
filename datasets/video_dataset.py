from json.scanner import NUMBER_RE
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import transforms

# DATASET_ROOT = '/media/amazon/F/henry/data/CHT/'
class VideoVPRDataset(Dataset):
    def __init__(self, ref_dir, query_dir):
        self.ref_paths = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.png')])
        self.query_paths = sorted([os.path.join(query_dir, f) for f in os.listdir(query_dir) if f.endswith('.png')])
        self.all_paths = self.ref_paths + self.query_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.num_references = len(self.ref_paths)
        self.num_queries = len(self.query_paths)

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img = Image.open(self.all_paths[idx]).convert('RGB')
        return (self.transform(img),)

    # def __getitem__(self, idx):
    #     img = Image.open(self.all_paths[idx]).convert('RGB')
    #     return self.transform(img)