import os
import h5py
import torch
import torch.utils.data as data
import psutil
import numpy as np


class PCamDatasets:
    train_x_path = "data/pcam/"\
                   "camelyonpatch_level_2_split_train_x.h5"
    train_y_path = "data/pcam/"\
                   "camelyonpatch_level_2_split_train_y.h5"
    train_meta_path = "data/pcam/camelyonpatch_level_2_split_train_meta.csv"
    train_paths = [train_x_path, train_y_path, train_meta_path]

    valid_x_path = "data/pcam/"\
                   "camelyonpatch_level_2_split_valid_x.h5"
    valid_y_path = "data/pcam/"\
                   "camelyonpatch_level_2_split_valid_y.h5"
    valid_meta_path = "data/pcam/camelyonpatch_level_2_split_valid_meta.csv"
    valid_paths = [valid_x_path, valid_y_path, valid_meta_path]

    test_x_path = "data/pcam/"\
                  "camelyonpatch_level_2_split_test_x.h5"
    test_y_path = "data/pcam/"\
                  "camelyonpatch_level_2_split_test_y.h5"
    test_meta_path = "data/pcam/camelyonpatch_level_2_split_test_meta.csv"
    test_paths = [test_x_path, test_y_path, test_meta_path]

    def __init__(self):
        super(PCamDatasets, self).__init__()
        self.train = PCamDataset(PCamDatasets.train_paths)
        self.valid = PCamDataset(PCamDatasets.valid_paths)
        self.test = PCamDataset(PCamDatasets.test_paths)


class PCamDataset(data.Dataset):
    def __init__(self, files):
        super(PCamDataset, self).__init__()

        x_path = files[0]
        x_file = h5py.File(x_path)
        self.data = x_file.get('x')

        y_path = files[1]
        y_file = h5py.File(y_path)
        self.target = y_file.get('y')

        meta_path = files[2]

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index, :, :, :]).float().permute(2, 0, 1),
                torch.from_numpy(self.target[index, :, :, :].ravel()).long())

    def __len__(self):
        return self.data.shape[0]
