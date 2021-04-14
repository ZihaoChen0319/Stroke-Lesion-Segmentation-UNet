import os
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np
from medpy.io import load, save
import pickle

from MyNetwork import Unet
from MyDataloader2D import get_dataloader, preprocess, img_normalize
from MyLosses import Dice


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device = 'cpu'

    train_path = './raw_data/train'
    test_path = './raw_data/test'
    batch_size = 16
    epochs = 500
    learning_rate = 1e-04

    model = Unet()
    model.to(device)

    loss_dice = Dice().loss
    loss_ce = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # # This line of code is used to process raw data, which should be run only once.
    # cnt = preprocess(train_path)

    img2 = pickle.load(open('./data/flair.pkl', 'rb'))
    img1 = pickle.load(open('./data/dwi.pkl', 'rb'))
    mask = pickle.load(open('./data/mask.pkl', 'rb'))
    all_data = img1, img2, mask

    for epoch in range(epochs):

        model.train()
        loss_value = [0, 0]

        train_dataloader = get_dataloader(batch_size, all_data, shuffle=True, num_workers=4)
        train_dataloader = tqdm(train_dataloader)

        for img_batch, mask_batch in train_dataloader:

            img_batch = img_batch.float().to(device)
            mask_batch = mask_batch.float().to(device).unsqueeze(1)
            img_batch = nnf.interpolate(img_batch, size=(128, 128), mode='bilinear')
            mask_batch = nnf.interpolate(mask_batch, size=(128, 128), mode='nearest')
            predict_batch = model(img_batch)

            loss = loss_dice(predict_batch, mask_batch) + loss_ce(predict_batch, mask_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value[0] += img_batch.size()[0]
            loss_value[1] += loss.detach().cpu().numpy() * img_batch.size()[0]
            train_dataloader.set_description("Train loss %f" % (loss_value[1] / loss_value[0]))

        scheduler.step()

        model.eval()
        loss_value = [0, 0]

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
            mask_test, h = load(mask_file)
            img_test1 = img_normalize(img_test1)
            img_test2 = img_normalize(img_test2)
            img_test = np.stack([img_test1, img_test2], axis=0)

            # Evaluate test performance in 3D.
            mask3d = torch.zeros((128, 128, img_test1.shape[2]))
            pred3d = torch.zeros((128, 128, img_test1.shape[2]))
            for i in range(img_test1.shape[2]):
                img_test_slice = torch.tensor(img_test[:, :, :, i]).float().to(device).unsqueeze(0)
                mask_test_slice = torch.tensor(mask_test[:, :, i]).float().to(device).unsqueeze(0).unsqueeze(0)
                img_test_slice = nnf.interpolate(img_test_slice, size=(128, 128), mode='bilinear')
                mask_test_slice = nnf.interpolate(mask_test_slice, size=(128, 128), mode='nearest')
                pred_test_slice = model(img_test_slice).detach()
                pred_test_slice[pred_test_slice >= 0.5] = 1
                pred_test_slice[pred_test_slice < 0.5] = 0
                pred3d[:, :, i] = pred_test_slice
                mask3d[:, :, i] = mask_test_slice
            pred3d = pred3d.unsqueeze(0).unsqueeze(0)
            mask3d = mask3d.unsqueeze(0).unsqueeze(0)
            pred3d = nnf.interpolate(pred3d, size=img_test1.shape, mode='trilinear')
            mask3d = nnf.interpolate(mask3d, size=img_test1.shape, mode='nearest')

            loss_value[0] += 1
            loss_value[1] += loss_dice(pred3d, mask3d).cpu().numpy()

        print('Epoch:', epoch + 1, ', test loss: %f' % (loss_value[1] / loss_value[0]))

        if (epoch + 1) % 10 == 0:
            torch.save(model, 'models/epoch_' + str(epoch+1) + '.pth')

