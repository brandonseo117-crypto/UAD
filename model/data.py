import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class MRITrainDataset(Dataset):
    def __init__(self, img_dir)
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