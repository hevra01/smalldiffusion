import torch
import csv
from torch.utils.data import Dataset

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N):
        t = tmin + torch.linspace(0, 1, N) * tmax
        self.vals = torch.stack([t * torch.cos(t) / tmax, t * torch.sin(t) / tmax]).T 

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]
    
class Swissroll_2(Dataset):
    def __init__(self, tmin, tmax, N):
        t = tmin + torch.linspace(0, 1, N) * tmax
        x = t * torch.cos(t)
        y = t * torch.sin(t)
        vals = torch.stack([x, y]).T

        # Normalize to [-1, 1]
        vals_min = vals.min(dim=0).values
        vals_max = vals.max(dim=0).values
        vals = 2 * (vals - vals_min) / (vals_max - vals_min) - 1

        # Scale to [-25, 25]
        self.vals = vals * 25

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

    

class FibonacciDataset(Dataset):
    def __init__(self, N):
        self.vals = self.generate_fibonacci(N)
    
    def generate_fibonacci(self, N):
        fibs = [0, 1]
        for i in range(2, N):
            fibs.append(fibs[-1] + fibs[-2])
        return torch.tensor(fibs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.vals)
    
    def __getitem__(self, i):
        return self.vals[i]
    

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15, delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]

# Mainly used to discard labels and only output data
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])
