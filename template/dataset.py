import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class TempDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if self.transform is None:
            x = self.X[index]
        else:
            x = self.transform(x)

        return x, self.y[index]