import sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from preprocess import train_valid_test_split

class ConvNetDataset(Dataset):
    def __init__(self, data, columns, desired_columns):
        self.data = data
        self.columns = columns

        assert len(self.data[0][0][0]) == len(columns)
        
        desired_indices = []
        for column_idx, column in enumerate(self.columns):
            if column in desired_columns:
                desired_indices.append(column_idx)

        self.features = []
        self.targets = []
        for one_data in self.data:
            chunk, target = one_data
            chunk = np.array(chunk)
            self.features.append(chunk.take(desired_indices, axis=-1))
            self.targets.append(target)
        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets,dtype=np.float32)
        
        self.features = np.expand_dims(self.features, axis=-1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "targets": self.targets[index]
        }

# class ConvNetDatasetDynamic(Dataset):
#     def __init__(self, array):
#         self.data = array
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         return self.data[index]

def get_dataloaders(desired_columns, train_ratio="9:1:2", window_size=7, batch_size=16, device="cpu"):
    columns, train, valid, test = train_valid_test_split(desired_columns=desired_columns, train_ratio, window_size)

    train_dataset = ConvNetDataset(data=train, columns=columns, desired_columns=desired_columns)
    valid_dataset = ConvNetDataset(data=valid, columns=columns, desired_columns=desired_columns)
    test_dataset = ConvNetDataset(data=test, columns=columns, desired_columns=desired_columns)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader