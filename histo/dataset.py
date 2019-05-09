"""Module contains classes and constants for loading datasets.
Main dataset is PatchCamelyon, available at https://github.com/basveeling/pcam."""
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

PCAM_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[178.69278045, 137.28123996, 176.36324185],
    #                      std=[59.91942025, 70.73932419, 54.28812066])
])
PCAM_VALID_TRANSFORM = PCAM_TRAIN_TRANSFORM
PCAM_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    #  transforms.Normalize(mean=[178.69278045, 137.28123996, 176.36324185],
    #                       std=[59.91942025, 70.73932419, 54.28812066])
])

PCAM_DATA_TRANSFORM = {
    'train': PCAM_TRAIN_TRANSFORM,
    'valid': PCAM_VALID_TRANSFORM,
    'test': PCAM_TEST_TRANSFORM
}


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

    def __init__(self, data_transforms=None, target_transforms=None):
        super(PCamDatasets, self).__init__()
        self.train = PCamDataset(PCamDatasets.train_paths,
                                 transform=data_transforms.get('train')
                                 if data_transforms is not None else None,
                                 target_transform=data_transforms.get('train')
                                 if target_transforms is not None else None)
        self.valid = PCamDataset(PCamDatasets.valid_paths,
                                 transform=data_transforms.get('valid')
                                 if data_transforms is not None else None,
                                 target_transform=target_transforms.get('valid')
                                 if target_transforms is not None else None)
        self.test = PCamDataset(PCamDatasets.test_paths,
                                transform=data_transforms.get('test')
                                if data_transforms is not None else None,
                                target_transform=target_transforms.get('test')
                                if target_transforms is not None else None)


class PCamDataset(data.Dataset):
    def __init__(self, files, transform=None, target_transform=None):
        super(PCamDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        x_path = files[0]
        x_file = h5py.File(x_path)
        self.data = x_file.get('x')

        y_path = files[1]
        y_file = h5py.File(y_path)
        self.target = y_file.get('y')
        self.num_examples = self.data.shape[0]
        # meta_path = files[2]

    def __getitem__(self, index):
        if index < 0 or index >= self.num_examples:
            raise IndexError
        img = Image.fromarray(self.data[index, :, :, :])
        target = torch.from_numpy(self.target[index, :, :, :].ravel()).float()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.num_examples
