import os
import torch
import shutil
import numpy as np
import random

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        # Read the original data as numpy arrays
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        sample = dict(image=image, mask=mask)

        if self.transform is not None:
            sample = self.transform(**sample)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        # Define full file paths
        trainval_path = os.path.join(self.root, "annotations", "trainval.txt")
        test_path = os.path.join(self.root, "annotations", "test.txt")

        # Read the contents of both files
        with open(trainval_path, "r") as f:
            trainval_lines = f.read().strip().split("\n")
        with open(test_path, "r") as f:
            test_lines = f.read().strip().split("\n")

        # Merge the lists and extract filenames (assuming the first field in each line is the filename)
        lines = trainval_lines + test_lines
        filenames = [line.split(" ")[0] for line in lines]

        n = len(filenames)
        train_end = int(0.8 * n)
        valid_end = int(0.9 * n)

        if self.mode == "train":
            return filenames[:train_end]
        elif self.mode == "valid":
            return filenames[train_end:valid_end]
        elif self.mode == "test":
            return filenames[valid_end:]
        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")


    @staticmethod
    def download(root):
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


# Define transformations for training using torchvision
def train_transform(image, mask):
    # Convert numpy arrays to PIL images
    image = Image.fromarray(image)
    mask = Image.fromarray(mask.astype(np.uint8))

    # Random horizontal flip
    if random.random() < 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)

    # RandomResizedCrop
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.8, 1.0), ratio=(0.75, 1.33)
    )
    image = F.resized_crop(image, i, j, h, w, (256, 256), interpolation=Image.BILINEAR)
    mask = F.resized_crop(mask, i, j, h, w, (256, 256), interpolation=Image.NEAREST)

    # Random affine transformation (simulating ShiftScaleRotate)
    angle = random.uniform(-30, 30)
    translate = (int(random.uniform(-0.2, 0.2) * 256), int(random.uniform(-0.2, 0.2) * 256))
    scale = random.uniform(0.8, 1.2)
    image = F.affine(
        image, angle=angle, translate=translate, scale=scale, shear=0, interpolation=Image.BILINEAR
    )
    mask = F.affine(
        mask, angle=angle, translate=translate, scale=scale, shear=0, interpolation=Image.NEAREST
    )

    # Color jitter (applied only to the image)
    color_jitter = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
    )
    image = color_jitter(image)

    # Convert to tensor and normalize
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )(image)
    # Convert mask to tensor, change to float and add channel dimension
    mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)

    return dict(image=image, mask=mask)
 

# Define transformations for validation and testing (only resize and normalize)
def valid_transform(image, mask):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask.astype(np.uint8))

    image = transforms.Resize((256, 256), interpolation=Image.BILINEAR)(image)
    mask = transforms.Resize((256, 256), interpolation=Image.NEAREST)(mask)

    image = transforms.ToTensor()(image)
    image = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )(image)
    # Convert mask to float tensor and add channel dimension
    mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)

    return dict(image=image, mask=mask)


def load_dataset(data_path, mode):
    if mode == "train":
        dataset = OxfordPetDataset(root=data_path, mode="train", transform=train_transform)
    elif mode == "valid":
        dataset = OxfordPetDataset(root=data_path, mode="valid", transform=valid_transform)
    elif mode == "test":
        dataset = OxfordPetDataset(root=data_path, mode="test", transform=valid_transform)
    return dataset
