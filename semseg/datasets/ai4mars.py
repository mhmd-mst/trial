import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple
import os
import sys

sys.path.append(os.getcwd())
from torchvision import transforms
from PIL import Image
import numpy as np
import random


def augmentation(image, label):
    if random.random() > 0.5:
        image, label = transforms.functional.hflip(image), transforms.functional.hflip(label)
    if random.random() > 0.5:
        image, label = transforms.functional.rotate(image, 0.2), transforms.functional.rotate(label, 0.2)
    return image, label


class ai4mars(Dataset):
    CLASSES = [
        'soil', 'bedrock', 'sand', 'bigrock'
    ]

    def __init__(self, path: str, split: str = 'train', scale: tuple = (512, 512), transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.path = path
        self.image_size = scale
        image_path = os.path.join(path, "images/edr/")
        image_name = [image.split(".")[0] for image in os.listdir(image_path)]
        if split == 'train':
            label_path = os.path.join(path, "labels/train/")
            label_name = [label.split(".")[0] for label in os.listdir(label_path)]
        else:
            label_path = os.path.join(path, "labels/test/masked-gold-min1-100agree")
            label_name = ["_".join(label.split("_")[:-1]) for label in os.listdir(label_path)]

        self.name_intersection = sorted(list(set(image_name) & set(label_name)))
        self.image_path = [os.path.join(image_path, image + ".JPG") for image in self.name_intersection]
        if split == 'train':
          self.label_path = [os.path.join(label_path, label + ".png") for label in self.name_intersection]
        else:
          self.label_path = [os.path.join(label_path, label + "_merged.png") for label in self.name_intersection]

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = Image.open(self.image_path[index]).convert("RGB")
        label = Image.open(self.label_path[index])
        if self.split == 'training':
            image, label = augmentation(image, label)
        image = transforms.Resize(self.image_size, Image.BILINEAR)(image)
        label = transforms.Resize(self.image_size, Image.NEAREST)(label)
        image = self.to_tensor(image)
        label = torch.from_numpy(np.asarray(label, dtype=np.float32))
        return image, label.long()


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample

    visualize_dataset_sample(ai4mars,
                             '/Users/almou/Desktop/Internship/Dataset/ai4mars-dataset-merged-0.1/ai4mars-dataset-merged-0.1/msl/')
