import os
import torch
from medpy.io import load, save
import torch.nn.functional as nnf
import numpy as np

from MyDataloader2D import open_file, img_normalize
from MyLosses import Dice


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device = 'cpu'

    test_path = './raw_data/test'
    save_path = './temp_result/test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = torch.load('./models/epoch_200.pth')
    model.to(device)
    model.eval()

    loss_dice = Dice().loss

    results = []
    test_folders = os.listdir(test_path)
    for folder in test_folders:
        folder_individual = test_path + '/' + folder + '/'
        all_data_individual = os.listdir(folder_individual)
        img_file1 = [name for name in all_data_individual if name[18:21] == 'DWI']
        img_file1 = folder_individual + img_file1[0] + '/' + img_file1[0] + '.nii'

        img_file2 = [name for name in all_data_individual if name[18:23] == 'Flair']
        img_file2 = folder_individual + img_file2[0] + '/' + img_file2[0] + '.nii'

        mask_file = [name for name in all_data_individual if name[15:17] == 'OT']
        mask_file = folder_individual + mask_file[0] + '/' + mask_file[0] + '.nii'

        img_test1, h = load(img_file1)
        img_test2, h = load(img_file2)
        img_test1 = img_normalize(img_test1)
        img_test2 = img_normalize(img_test2)

        img_test = np.stack([img_test1, img_test2], axis=0)
        mask_test, h = load(mask_file)

        mask3d = torch.zeros((128, 128, img_test.shape[-1]))
        pred3d = torch.zeros((128, 128, img_test.shape[-1]))
        for i in range(img_test.shape[-1]):
            img_test_slice = torch.tensor(img_test[:, :, :, i]).float().to(device).unsqueeze(0)
            mask_test_slice = torch.tensor(mask_test[:, :, i]).float().to(device).unsqueeze(0).unsqueeze(0)
            img_test_slice = nnf.interpolate(img_test_slice, size=(128, 128), mode='bilinear')
            mask_test_slice = nnf.interpolate(mask_test_slice, size=(128, 128), mode='nearest')
            pred_test_slice = model(img_test_slice).detach()
            pred3d[:, :, i] = pred_test_slice
            mask3d[:, :, i] = mask_test_slice
        pred3d = pred3d.unsqueeze(0).unsqueeze(0)
        mask3d = mask3d.unsqueeze(0).unsqueeze(0)
        pred3d = nnf.interpolate(pred3d, size=img_test1.shape, mode='trilinear')
        mask3d = nnf.interpolate(mask3d, size=img_test1.shape, mode='nearest')
        pred3d[pred3d >= 0.5] = 1
        pred3d[pred3d < 0.5] = 0

        dice_single = loss_dice(pred3d, mask3d).cpu().numpy()
        result_single = (folder, dice_single)
        results.append(result_single)
        print(result_single)

        save(pred3d.cpu().numpy().squeeze(), save_path + '/' + folder + '.nii.gz', hdr=h)

