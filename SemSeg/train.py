# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:25:12 2021

@author: GREEN&LYC
"""

# from data_process import train_data,test_data
from test_resunet import ResNet18UNet

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import segmentation_models_pytorch as smp

from PIL import Image
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import seaborn as sns

import time
from tqdm.notebook import tqdm

# summary(model, input_size=(3, 224, 224))



def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(nn.functional.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(train_loader, test_loader, epochs=10):
    # Load model
    # model = smp.Unet('resnet18', encoder_weights='imagenet', classes=13, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=13, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    
    # model = ResNet18UNet(13).to(device)
    model = model.to(device)
    weight_list=[0.8, 17.8, 9.9, 0.13, 0.92, 0.26, 0.72, 1.43, 1, 1.22, 0.56, 9.822, 16.86]
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight_list)).cuda().float())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 ,weight_decay=1e-4)
    # scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
    #                                        factor=0.1, patience=10, threshold=0.0001, \
    #                                            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, \
    #                                                verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs,
                                            steps_per_epoch=len(train_loader))
    # Train model
    model.train()
    running_loss = 0
    # print_every = 10
    train_acc,test_acc = [],[]
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n--------------------- ")
        #training loop
        steps = 0
        train_accuracy=0
        running_loss =0
        
        for inputs, labels in train_loader:
            steps += 1 
            # print(steps)
            # try:
            # model.cuda()
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer.zero_grad()
            # logps = model.forward(inputs)
            # loss = loss_fn(logps, labels)
            pred = model(inputs)
            # pred = pred.cpu().data.numpy()
            loss = loss_fn(pred, labels)
            
            train_accuracy += pixel_accuracy(pred, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # model.cpu()
            # loss.backward()
            # optimizer.step()
            running_loss += loss.item()
            # print(len(train_loader),type(train_accuracy),type(running_loss))
            # except Exception as e:
            #     print(e)

            # if steps % print_every == 0:
        test_loss = 0
        test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for test_in, test_lbl in test_loader:
                test_in, test_lbl = test_in.to(device), test_lbl.to(device)
                
                
                pred = model(test_in)
                # pred = pred.cpu().data.numpy()
                test_accuracy += pixel_accuracy(pred, test_lbl)
                test_loss += loss_fn(pred, test_lbl).item()
                    
            
        train_acc.append(train_accuracy/len(train_loader))
        test_acc.append(test_accuracy/ len(test_loader))
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        print(#f"[{len(train_loader)*len(inputs):5d}/{len(train_loader.dataset):5d}]"
              f"Train loss: {running_loss / len(train_loader):.3f}.. "
              f"Train accuracy: {train_accuracy / len(train_loader)*100:.3f}%.. "
              f"Test loss: {test_loss / len(test_loader):.3f}.. "
              f"Test accuracy: {test_accuracy / len(test_loader)*100:.3f}%")
        # print(optimizer.state_dict()['param_groups'])
        print(get_lr(optimizer))
        # model.train()
        scheduler.step(test_loss)
        torch.cuda.empty_cache()
                # running_loss = 0
                # train_accuracy = 0
                
    torch.save(model, model.name+'wCE'+'.pt')

    # Plot losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    # plt.savefig('materials_resnet.jpg')

# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.train()
#     train_accuracy=0
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#         train_accuracy += pixel_accuracy(pred, y)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 10 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#     train_accuracy /= num_batches
#     print(f"Train Error: \n train_accuracy: {(100*train_accuracy ):>0.1f}%")

# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, test_accuracy = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             test_accuracy += pixel_accuracy(pred, y)
#             # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     test_accuracy /= num_batches
#     print(f"Test Error: \n test_accuracy: {(100*test_accuracy ):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__' :
    from data_process import DataGenerator,train_label_dir,train_img_dir,test_label_dir, test_img_dir,\
        get_image_pixel_mean,get_image_pixel_std,HVRotate90,Rotate90,H,V,HV   # ,train_data1,test_data
    # from local_data_process import DataGenerator,train_label_dir,train_img_dir,test_label_dir, test_img_dir
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = 'cpu'
    print(f"Using {device} device")
    batch_size = 8
    epochs = 10
    
    mean = get_image_pixel_mean(train_img_dir)
    std = get_image_pixel_std(train_img_dir, mean)

    train_data1 = DataGenerator(train_label_dir,train_img_dir, mean, std)
    train_data2 = DataGenerator(train_label_dir,train_img_dir, mean, std,H)
    train_data3 = DataGenerator(train_label_dir,train_img_dir, mean, std,V)
    train_data4 = DataGenerator(train_label_dir,train_img_dir, mean, std,HV)
    train_data5 = DataGenerator(train_label_dir,train_img_dir, mean, std,Rotate90)
    train_data6 = DataGenerator(train_label_dir,train_img_dir, mean, std,HVRotate90)

    train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6])

    # test_data = DataGenerator(test_label_dir, test_img_dir)
    # train_data = DataGenerator(train_label_dir,train_img_dir,mean,std)
    test_data = DataGenerator(test_label_dir, test_img_dir,mean,std)
  
    train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size = batch_size, shuffle =True)
    
    train_model(train_dataloader, test_dataloader,epochs)
    
    
    
    
    # epochs = 10
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # print("Done!")
