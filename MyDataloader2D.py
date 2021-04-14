import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from medpy.io import load, save
import pickle
import tqdm
import torch


def preprocess(path):
    """Transform the 3D image to 2D slices, do normalization and save."""
    all_folders = os.listdir(path)
    flair = []
    dwi = []
    seg = []
    cnt = 0
    for folder in all_folders:
        folder_individual = path + '/' + folder + '/'
        all_data_individual = os.listdir(folder_individual)

        img_file1 = [name for name in all_data_individual if name[18:21] == 'DWI']
        img_file1 = folder_individual + img_file1[0] + '/' + img_file1[0] + '.nii'

        img_file2 = [name for name in all_data_individual if name[18:23] == 'Flair']
        img_file2 = folder_individual + img_file2[0] + '/' + img_file2[0] + '.nii'

        mask_file = [name for name in all_data_individual if name[15:17] == 'OT']
        mask_file = folder_individual + mask_file[0] + '/' + mask_file[0] + '.nii'

        img1, h = load(img_file1)
        img1 = img_normalize(img1)
        img2, h = load(img_file2)
        img2 = img_normalize(img2)
        mask, h = load(mask_file)

        img_shape = img1.shape
        for i in range(img_shape[-1]):
            dwi.append(np.squeeze(img1[:, :, i]))
            flair.append(np.squeeze(img2[:, :, i]))
            seg.append(np.squeeze(mask[:, :, i]))
            cnt = cnt+1

    pickle.dump(dwi, open('./data/dwi.pkl', 'wb'))
    pickle.dump(flair, open('./data/flair.pkl', 'wb'))
    pickle.dump(seg, open('./data/mask.pkl', 'wb'))
    return cnt


def img_normalize(img):
    """z-score normalization."""
    minimum_gray = np.min(img)
    brain = img[(img - minimum_gray) > 1e-4]
    img[(img - minimum_gray) > 1e-4] = (brain - np.mean(brain)) / np.std(brain)
    #img = (img - min([0, np.min(img)])) / (np.max(img) - min([0, np.min(img)]))
    return img


def open_file(index, input_shape=None):
    """Open .nii file and padding to input_shape."""
    mask, h = load(index[2])
    img1, h = load(index[0])
    img2, h = load(index[1])

    img1 = img_normalize(img1)
    img2 = img_normalize(img2)

    if not input_shape:
        input_shape = img1.shape
    pads = [input_shape[n] - img1.shape[n] for n in range(len(img1.shape) - 1)]
    pad_amount = [(np.floor(pad/2).astype(int), np.ceil(pad/2).astype(int)) for pad in pads]
    img1 = np.pad(np.squeeze(img1[:, :, index[3]]), pad_amount, mode='minimum')
    img2 = np.pad(np.squeeze(img2[:, :, index[3]]), pad_amount, mode='minimum')
    mask = np.pad(np.squeeze(mask[:, :, index[3]]), pad_amount, mode='minimum')

    return img1, img2, mask, h



class TestDataset(Dataset):
    def __init__(self, all_data):
        self.data = all_data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        img1 = self.data[0][item]
        img2 = self.data[1][item]
        img = np.stack([img1, img2], axis=0)
        mask = self.data[2][item]
        return img, mask / 1.0


def get_dataloader(batch_size, all_slice, shuffle=True, num_workers=4):

    dataset = TestDataset(all_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)#3533/16=221（组）
    return dataloader


