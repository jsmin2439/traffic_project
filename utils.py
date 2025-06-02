# utils.py
import numpy as np
import torch
from torch.utils.data import Dataset

class TrafficWindowDataset(Dataset):
    """all_X, all_Y → PyTorch Dataset"""

    def __init__(self, X_np, Y_np):
        self.X = torch.from_numpy(X_np).float()  # (N,12,1370,9)
        self.Y = torch.from_numpy(Y_np).float()  # (N,1370,8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].permute(2, 0, 1)   # (9,12,1370)
        y = self.Y[idx].permute(1, 0)      # (8,1370)
        return x, y

class EarlyStopping:
    """patience 만큼 연속으로 개선이 없으면 stop"""
    def __init__(self, patience=5):
        self.patience = patience
        self.counter  = 0
        self.best     = np.inf
        self.stop     = False

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True