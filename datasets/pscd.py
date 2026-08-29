import glob
import os

import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tvff

_file_path = os.path.split(os.path.realpath(__file__))[0]


class PSCD:

    def __init__(self, root, mode="train", img_size=256):

        assert mode in {"train", "test", "val"}

        self.mode = mode
        self.root = root
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Resize((img_size, img_size), antialias=True),  # Resize first
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize((img_size, img_size), antialias=True),  # Resize masks            
        ])

        # for simplicity, this class do not perform any security check
        # the assumptions are:
        # 1. [t0, t1, mask_t0, mask_t1] folders must contain in root
        # 2. all files in [t0, t1, mask_t0, mask_t1] use the exactly
        #    same file name.
        self._t0 = os.path.join(self.root, "t0")
        self._t1 = os.path.join(self.root, "t1")
        self._mask_t0 = os.path.join(self.root, "mask")
        self._mask_t1 = os.path.join(self.root, "mask_t1")

        filenames = glob.glob(os.path.join(self._t0, "*.png"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)
        filenames = np.array(filenames)

        path = os.path.join(_file_path, f"indices/pscd.{mode}.index")
        with open(path, "r") as fd:
            indices = fd.read().splitlines()

        I = np.searchsorted(filenames, indices)
        assert np.all(filenames[I] == np.array(indices))

        self._filenames = filenames[I]
        self.hflip_prob = 0.5

    @property
    def filenames(self):
        return self._filenames

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):

        filename = self._filenames[idx]
        t0 = os.path.join(self._t0, filename)
        t1 = os.path.join(self._t1, filename)
        t0_mask = os.path.join(self._mask_t0, filename)
        t1_mask = os.path.join(self._mask_t1, filename)

        t0_image = Image.open(t0).convert("RGB")
        t1_image = Image.open(t1).convert("RGB")
        t0_mask = Image.open(t0_mask).convert("L")
        t1_mask = Image.open(t1_mask).convert("L")

        t0_image = np.array(t0_image) / 255.0
        t1_image = np.array(t1_image) / 255.0
        t0_mask = np.array(t0_mask) / 255.0
        t1_mask = np.array(t1_mask) / 255.0

        t0_mask = t0_mask > 0.0
        t1_mask = t1_mask > 0.0

        t0_image = t0_image.astype(np.float32)
        t1_image = t1_image.astype(np.float32)
        t0_mask = t0_mask.astype(np.float32)
        t1_mask = t1_mask.astype(np.float32)

        return t0_image, t1_image, t0_mask, t1_mask

    @property
    def figsize(self):
        return np.array([224, 1024])


class CroppedPSCD(PSCD):

    def __init__(
        self,
        root,
        mode="train",
        crop_num=15,
        use_mask_t0=True,
        use_mask_t1=False,
    ):

        if crop_num <= 1:
            raise ValueError("crop_num must be greater than 1")

        super().__init__(root, mode)

        width = super().figsize[1]

        # hardcode 224 as the width of cropped image
        stride = (width - 224) // (crop_num - 1)
        cropped_length = 224 + (crop_num - 1) * stride
        cropped_prefix = (width - cropped_length) // 2

        self.stride = stride
        self.crop_num = crop_num
        self.cropped_length = cropped_length
        self.cropped_prefix = cropped_prefix

        self.use_mask_t0 = use_mask_t0
        self.use_mask_t1 = use_mask_t1

        self._crop_filenames = None

    @property
    def original_filenames(self):
        return np.repeat(self._filenames, self.crop_num)

    @property
    def filenames(self):

        if self._crop_filenames is not None:
            return self._crop_filenames

        N = len(self._filenames)
        names = self.original_filenames
        index = np.arange(self.crop_num)[None, :]
        index = np.repeat(index, N, axis=0)
        index = index.flatten()

        assert len(index) == len(names)

        filenames = []
        for name, ind in zip(names, index):
            name = name.replace(".png", f".{ind}.png")
            filenames.append(name)
        self._crop_filenames = np.array(filenames)

        return self._crop_filenames

    def __len__(self):
        return len(self._filenames) * self.crop_num

    def __getitem__(self, idx):

        id_image = idx // self.crop_num
        t0_image, t1_image, t0_mask, t1_mask = super().__getitem__(id_image)

        # remove prefix and suffix from original image
        cropped_area = slice(
            self.cropped_prefix,
            self.cropped_prefix + self.cropped_length,
            None,
        )

        t0_image = t0_image[:, cropped_area, :]
        t1_image = t1_image[:, cropped_area, :]
        t0_mask = t0_mask[:, cropped_area]
        t1_mask = t1_mask[:, cropped_area]

        # cropped image to 224 * 224
        id_crop = idx % self.crop_num

        cropped_area = slice(
            self.stride * id_crop,
            self.stride * id_crop + 224,
            None,
        )

        t0_image = t0_image[:, cropped_area, :]
        t1_image = t1_image[:, cropped_area, :]
        t0_mask = t0_mask[:, cropped_area]
        t1_mask = t1_mask[:, cropped_area]

        t0_image = self.transform(t0_image)
        t1_image = self.transform(t1_image)
        
        t0_mask = self.mask_transform(t0_mask)
        
        if self.mode == "train" and np.random.random() < self.hflip_prob:
            t0_image = tvff.hflip(t0_image)
            t1_image = tvff.hflip(t1_image)
            t0_mask = tvff.hflip(t0_mask)
            
        output = t0_image, t1_image, t0_mask

        # if self.use_mask_t0:
        #     t0_mask = self.mask_transform(t0_mask)
        #     output += (t0_mask,)

        # if self.use_mask_t1:
        #     output += (t1_mask,)

        return output

    @property
    def figsize(self):
        return np.array([224, 224])


class DiffViewPSCD(CroppedPSCD):

    def __init__(
        self,
        root,
        mode="train",
        crop_num=15,
        adjacent_distance=1,
    ):

        super().__init__(
            root,
            mode=mode,
            crop_num=crop_num,
            use_mask_t0=True,
            use_mask_t1=False,
        )

        if adjacent_distance * self.stride >= 224:
            x = 224 // self.stride
            msg = f"adjacent_distance must be within [{-x}, {x})"
            raise ValueError(msg)

        # indices for every t0
        start = 0 if adjacent_distance >= 0 else -adjacent_distance
        end = 0 if adjacent_distance < 0 else adjacent_distance

        start_ind = np.arange(len(self._filenames)) * crop_num

        indices = np.arange(start, crop_num - end)
        indices = indices[None, :] + start_ind[:, None]
        indices = indices.flatten()

        self.adjacent_distance = adjacent_distance
        self.indices = indices

        self._diff_crop_filenames = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx = self.indices[idx]
        t0_image, _, t0_mask = super().__getitem__(idx)
        _, t1_image, _ = super().__getitem__(idx + self.adjacent_distance)

        if self.adjacent_distance >= 0:
            t0_mask[:, : self.adjacent_distance * self.stride] = 0.0
        else:
            t0_mask[:, self.adjacent_distance * self.stride :] = 0.0

        return t0_image, t1_image, t0_mask

    @property
    def filenames(self):

        if self._diff_crop_filenames is not None:
            return self._diff_crop_filenames

        names = super().filenames

        t0_names = names[self.indices]
        t1_names = names[self.indices + self.adjacent_distance]

        names = []
        for t0_name, t1_name in zip(t0_names, t1_names):
            t0_name = t0_name.replace(".png", "")
            t1_name = t1_name.replace(".png", "")

            name, t0_ind = t0_name.split(".")
            _, t1_ind = t1_name.split(".")
            name = f"{name}.{t0_ind}-{t1_ind}.png"
            names.append(name)
        self._diff_crop_filenames = np.array(names)
        return self._diff_crop_filenames
