import numpy as np
import torch
from pathlib import Path

mins = []
maxs = []

# Convert generator to a list so it can be reused and measured
folder_path = Path(r"C:\Users\Owner\Downloads\inference\CN")
scans = list(folder_path.iterdir())
length = len(scans)

print(f"Found {length} files to process...")

for scan in scans:
    payload = torch.load(scan, weights_only=False)
    tensor = payload["tensor"]
    mins.append(tensor.min().item())
    maxs.append(tensor.max().item())

mins = np.array(mins)
maxs = np.array(maxs)

print("--- MINIMUM VALUES ACROSS TEST SET ---")
print(f"Mean Min:   {np.mean(mins):.4f}")
print(f"Median Min: {np.median(mins):.4f}")
print(f"Std Min:    {np.std(mins):.4f}")
print(f"Range Min:  [{np.min(mins):.4f}, {np.max(mins):.4f}]")

print("\n--- MAXIMUM VALUES ACROSS TEST SET ---")
print(f"Mean Max:   {np.mean(maxs):.4f}")
print(f"Median Max: {np.median(maxs):.4f}")
print(f"Std Max:    {np.std(maxs):.4f}")
print(f"Range Max:  [{np.min(maxs):.4f}, {np.max(maxs):.4f}]")