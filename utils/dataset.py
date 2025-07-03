from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2

class ImgAugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentor=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentor = augmentor

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.augmentor:
            image_np = self.augmentor(image=image_np)

        image_np = cv2.resize(image_np, (500, 500))
        image = Image.fromarray(image_np.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label
