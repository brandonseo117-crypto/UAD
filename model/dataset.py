import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MRITrainDataset(Dataset): # unlabeled for unsupervised anomaly detection
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_paths = [img_path for img_path in Path(self.img_dir).iterdir()]
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        metadata = torch.load(img_path, weights_only=True)
        x_tensor = metadata['tensor']
        return x_tensor

train_dataset = MRITrainDataset(img_dir='sample_path')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

class MRITestDataset(Dataset): # has labels but aren't directly used in inference, used for later data analysis
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)
        self.sorted_dir = sorted(self.img_dir.iterdir())
        self.labels = {subfolder.name: i for i, subfolder in enumerate(self.sorted_dir)}
        self.payload_paths = []
        for subfolder in self.sorted_dir:
            for payload in subfolder.iterdir():
                self.payload_paths.append(payload)
    def __len__(self):
        return len(self.payload_paths)
    def __getitem__(self, idx):
        payload_path = self.payload_paths[idx]
        label = self.labels[payload_path.parent.name]
        metadata = torch.load(payload_path, weights_only=False)
        x_tensor = metadata['tensor']
        vis_date = metadata['scan_date']
        pt_id = metadata['ptid']

        return x_tensor, label, vis_date, pt_id

test_dataset = MRITestDataset('sample_path')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
