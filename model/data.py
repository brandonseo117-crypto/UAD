import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets import ImageFolder
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

train_loader = DataLoader(MRITrainDataset(img_dir), batch_size=1, shuffle=True)

test_dataset = ImageFolder(root='path/to/test_data', transform=None)


