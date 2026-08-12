import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from pathlib import Path

class MRITrainDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
    def __len__(self):
        return len(self.img_dir)
    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        metadata = torch.load(img_path)
        x_tensor = metadata['tensor']
        return x_tensor

test_dataset = datasets.ImageFolder('dataset_path')
train_dataset = MRITrainDataset(img_dir='sample_path')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class MRITestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)
        self.sorted_dir = sorted(self.img_dir.iterdir())
        self.labels = [subfolder.name for subfolder in self.sorted_dir]
        self.img_paths = []
        for subfolder in self.sorted_dir:
            for payload in subfolder.iterdir():
                self.img_paths.append(payload)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = Path(img_path).parent.name
        metadata = torch.load(img_path)
        x_tensor = metadata['tensor']
        vis_date = metadata['vis_date']
        pt_id = metadata['pt_id']

        return x_tensor, label, vis_date, pt_id