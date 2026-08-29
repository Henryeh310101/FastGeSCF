from json.scanner import NUMBER_RE
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms as T

# DATASET_ROOT = '/media/amazon/F/henry/data/CHT/'
class VPR_Dataset(Dataset):
    def __init__(self, ref_dir=None, query_dir=None, img_size=504):
        self.input_transform = self.transform(img_size)

        def numeric_sort(file_list):
            return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
        
        self.dbImages = numeric_sort(os.listdir(ref_dir))
        self.qImages = numeric_sort(os.listdir(query_dir))

        # Store full paths for loading images
        self.ref_dir = ref_dir
        self.query_dir = query_dir

        # Combined list of all images (ref + query)
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    def transform(self, image_size):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
    
    def __getitem__(self, index):
        if index < self.num_references:
            img_path = os.path.join(self.ref_dir, self.dbImages[index])
        else:
            img_path = os.path.join(self.query_dir, self.qImages[index - self.num_references])

        img = Image.open(img_path).convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                f.write(f"{q} {db}\n")
                
    def predictions_only(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i])
                db = Path(self.dbImages[preds[i][0]])
                f.write(f"{db}\n")
                
    def get_queries(self, selected_idx):
        # Directly select images based on indices
        q_images_paths = [self.qImages[i] for i in selected_idx]
        q_images = []

        for img_path in q_images_paths:
            img_path = self.dataset_root + img_path
            img = Image.open(img_path)
            q_images.append(img)
        return q_images

    def get_references(self, selected_idx):
        # Directly select images based on indices
        ref_images_paths = [self.dbImages[i] for i in selected_idx]
        ref_images = []

        for img_path in ref_images_paths:
            img_path = self.dataset_root + img_path
            img = Image.open(img_path)
            ref_images.append(img)
        return ref_images