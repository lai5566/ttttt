# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np


resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class Dataset_(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 crop_long_edge=False,
                 resize_size=None,
                 resizer="lanczos",
                 random_flip=False,
                 normalize=True,
                 hdf5_path=None,
                 load_data_in_memory=False,
                 translation_mode="N/A",
                 domain_A_path="trainA",
                 domain_B_path="trainB"):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.random_flip = random_flip
        self.normalize = normalize
        self.hdf5_path = hdf5_path
        self.load_data_in_memory = load_data_in_memory
        self.translation_mode = translation_mode
        self.domain_A_path = domain_A_path
        self.domain_B_path = domain_B_path
        self.trsf_list = []

        if self.hdf5_path is None:
            if crop_long_edge:
                self.trsf_list += [CenterCropLongEdge()]
            if resize_size is not None and resizer != "wo_resize":
                self.trsf_list += [transforms.Resize(resize_size, interpolation=resizer_collection[resizer])]
        else:
            self.trsf_list += [transforms.ToPILImage()]

        if self.random_flip:
            self.trsf_list += [transforms.RandomHorizontalFlip()]

        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        else:
            self.trsf_list += [transforms.PILToTensor()]

        self.trsf = transforms.Compose(self.trsf_list)

        self.load_dataset()

    def load_dataset(self):
        if self.hdf5_path is not None:
            with h5.File(self.hdf5_path, "r") as f:
                data, labels = f["imgs"], f["labels"]
                self.num_dataset = data.shape[0]
                if self.load_data_in_memory:
                    print("Load {path} into memory.".format(path=self.hdf5_path))
                    self.data = data[:]
                    self.labels = labels[:]
            return

        # Image-to-image translation modes
        if self.translation_mode == "unpaired":
            # CycleGAN: Load from separate domain directories
            path_A = os.path.join(self.data_dir, self.domain_A_path)
            path_B = os.path.join(self.data_dir, self.domain_B_path)
            self.data_A = ImageFolder(root=path_A)
            self.data_B = ImageFolder(root=path_B)
            self.num_A = len(self.data_A)
            self.num_B = len(self.data_B)
            print(f"Loaded unpaired data: {self.num_A} images from domain A, {self.num_B} images from domain B")
            return

        elif self.translation_mode == "paired":
            # Pix2Pix: Load from single directory with concatenated images
            mode = "train" if self.train == True else "valid"
            root = os.path.join(self.data_dir, mode)
            self.data = ImageFolder(root=root)
            print(f"Loaded paired data: {len(self.data)} image pairs")
            return

        elif self.translation_mode == "ct_mask":
            # StyleGAN3 Dual: Load paired CT and mask from separate directories
            # Expected structure:
            #   data_dir/
            #   ├── ct/train/ or ct/
            #   │   └── class0/
            #   │       ├── ct_001.png
            #   │       └── ...
            #   └── masks/train/ or masks/
            #       └── class0/
            #           ├── ct_001.png  # Same filename as CT
            #           └── ...

            mode = "train" if self.train == True else "valid"

            # Try with mode subdirectory first, then without
            ct_path_with_mode = os.path.join(self.data_dir, "ct", mode)
            mask_path_with_mode = os.path.join(self.data_dir, "masks", mode)
            ct_path_no_mode = os.path.join(self.data_dir, "ct")
            mask_path_no_mode = os.path.join(self.data_dir, "masks")

            if os.path.exists(ct_path_with_mode):
                ct_root = ct_path_with_mode
                mask_root = mask_path_with_mode
            elif os.path.exists(ct_path_no_mode):
                ct_root = ct_path_no_mode
                mask_root = mask_path_no_mode
            else:
                raise ValueError(f"CT data not found at {ct_path_with_mode} or {ct_path_no_mode}")

            # Load CT and mask datasets
            try:
                self.data_ct = ImageFolder(root=ct_root)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CT images from {ct_root}\n"
                    f"Error: {e}\n"
                    f"Please ensure the directory structure is:\n"
                    f"  {ct_root}/class0/*.png (or .jpg)"
                ) from e

            try:
                self.data_mask = ImageFolder(root=mask_root)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load mask images from {mask_root}\n"
                    f"Error: {e}\n"
                    f"Please ensure the directory structure is:\n"
                    f"  {mask_root}/class0/*.png (or .jpg)"
                ) from e

            # STRICT VALIDATION: CT and mask counts MUST match exactly
            if len(self.data_ct) != len(self.data_mask):
                raise ValueError(
                    f"CT-Mask count mismatch!\n"
                    f"  CT images: {len(self.data_ct)} (from {ct_root})\n"
                    f"  Mask images: {len(self.data_mask)} (from {mask_root})\n"
                    f"REQUIREMENT: Every CT image MUST have a corresponding mask with the SAME filename.\n"
                    f"Please check:\n"
                    f"  1. File counts match: ls {ct_root}/class0/ | wc -l\n"
                    f"  2. Filenames match: diff <(ls {ct_root}/class0/) <(ls {mask_root}/class0/)\n"
                    f"See CT_MASK_DATA_GUIDE.md for details."
                )

            # STRICT VALIDATION: Check if datasets are empty
            if len(self.data_ct) == 0:
                raise ValueError(
                    f"No CT images found in {ct_root}\n"
                    f"Please ensure images are placed in: {ct_root}/class0/*.png"
                )

            self.num_pairs = len(self.data_ct)
            print(f"✅ Loaded CT-Mask paired data: {self.num_pairs} pairs")
            print(f"   CT path: {ct_root}")
            print(f"   Mask path: {mask_root}")
            return

        # Standard GAN mode
        if self.data_name == "CIFAR10":
            self.data = CIFAR10(root=self.data_dir, train=self.train, download=True)

        elif self.data_name == "CIFAR100":
            self.data = CIFAR100(root=self.data_dir, train=self.train, download=True)
        else:
            mode = "train" if self.train == True else "valid"
            root = os.path.join(self.data_dir, mode)
            self.data = ImageFolder(root=root)

    def _get_hdf5(self, index):
        with h5.File(self.hdf5_path, "r") as f:
            return f["imgs"][index], f["labels"][index]

    def __len__(self):
        if self.translation_mode == "unpaired":
            # For CycleGAN, use max of both domains
            return max(self.num_A, self.num_B)
        elif self.translation_mode == "ct_mask":
            # For StyleGAN3 Dual, use number of CT-mask pairs
            return self.num_pairs
        elif self.hdf5_path is None:
            num_dataset = len(self.data)
        else:
            num_dataset = self.num_dataset
        return num_dataset

    def __getitem__(self, index):
        # Unpaired mode (CycleGAN)
        if self.translation_mode == "unpaired":
            # Get image from domain A
            index_A = index % self.num_A
            img_A, label_A = self.data_A[index_A]

            # Get image from domain B (randomized for unpaired)
            index_B = random.randint(0, self.num_B - 1)
            img_B, label_B = self.data_B[index_B]

            # Return: (img_A, label_A, img_B, label_B)
            return self.trsf(img_A), int(label_A), self.trsf(img_B), int(label_B)

        # CT-Mask paired mode (StyleGAN3 Dual)
        elif self.translation_mode == "ct_mask":
            # Get CT image
            img_ct, label_ct = self.data_ct[index]

            # Get corresponding mask image
            img_mask, label_mask = self.data_mask[index]

            # Apply transformations
            # Note: For masks, we need to be careful with normalization
            # CT: normalized to [-1, 1]
            # Mask: normalized to [0, 1] (binary mask)

            # Transform CT normally
            ct_transformed = self.trsf(img_ct)

            # Transform mask without color normalization
            # Create mask-specific transform
            mask_trsf_list = []
            if self.random_flip:
                # Apply same flip as CT (need to ensure consistency)
                # For now, we'll handle this in the training loop
                pass
            mask_trsf_list += [transforms.ToTensor()]  # [0, 1]
            # Convert to grayscale if mask is RGB
            if img_mask.mode == 'RGB':
                mask_trsf_list = [transforms.Grayscale(num_output_channels=1)] + mask_trsf_list

            mask_trsf = transforms.Compose(mask_trsf_list)
            mask_transformed = mask_trsf(img_mask)

            # Return: (ct_image, mask_image, label)
            # label is typically the same for both (class label)
            return ct_transformed, mask_transformed, int(label_ct)

        # Paired mode (Pix2Pix)
        elif self.translation_mode == "paired":
            img, label = self.data[index]

            # Split horizontally concatenated image [A|B]
            w, h = img.size
            w2 = int(w / 2)
            img_A = img.crop((0, 0, w2, h))
            img_B = img.crop((w2, 0, w, h))

            # Return: (img_A, img_B, label)
            return self.trsf(img_A), self.trsf(img_B), int(label)

        # Standard mode (regular GAN)
        else:
            if self.hdf5_path is None:
                img, label = self.data[index]
            else:
                if self.load_data_in_memory:
                    img, label = self.data[index], self.labels[index]
                else:
                    img, label = self._get_hdf5(index)
            return self.trsf(img), int(label)
