# Custom Dataset class for handling the specific image format

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, dataset, labels, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert idx to a Python int if it's a numpy.int64
        if isinstance(idx, np.int64):
            idx = int(idx)
        item = self.dataset[idx]
        image_data = item['image'].convert("RGB")  # Convert to RGB if necessary
        if self.transform:
            image = self.transform(image_data)
        label = self.labels.index(item['mana_color'])
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label
