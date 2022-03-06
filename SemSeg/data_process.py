# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:04:58 2021

@author: GREEN&LYC
"""
import os
import glob
import torch
import urllib
import math
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import albumentations as A
import random

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import seaborn as sns

train_img_dir = r"//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train"
train_label_dir = r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train_label'
test_img_dir = r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/test'
test_label_dir = r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/test_label'

# train_img_dir = r"/z/00INTERN/LiYuanchen/landuse_diff/data/train"
# train_label_dir = r'/z/00INTERN/LiYuanchen/landuse_diff/data/train_label'
# test_img_dir = r'/z/00INTERN/LiYuanchen/landuse_diff/data/test'
# test_label_dir = r'/z/00INTERN/LiYuanchen/landuse_diff/data/test_label'

# input_image = cv2.imread(imgpath)
# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# input_gt = cv2.imread(gtpath)
# input_gt = cv2.cvtColor(input_gt, cv2.COLOR_BGR2RGB)

# m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=m, std=s),
# ])

# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)
def get_image_pixel_mean(img_dir):
    """求数据集图像的R、G、B均值
    args: img_dir:
          img_list:
          img_size:
    """
    R_sum = 0
    G_sum = 0
    B_sum = 0
    count = 0
    # 循环读取所有图片
    for img_path in glob.glob(img_dir + '/*.jpg'):
        # img_path = os.path.join(img_dir, img_name)
        if not os.path.isdir(img_path):
            # image = cv2.imread(img_path)
            image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image =image/255.0
            # image = cv2.resize(image, (img_size, img_size))      # <class 'numpy.ndarray'>
            R_sum += image[:, :, 0].mean()
            G_sum += image[:, :, 1].mean()
            B_sum += image[:, :, 2].mean()
            count += 1
    R_mean = R_sum / count 
    G_mean = G_sum / count 
    B_mean = B_sum / count 
    print('R_mean:{}, G_mean:{}, B_mean:{}'.format(R_mean,G_mean,B_mean))
    RGB_mean = [R_mean, G_mean, B_mean]
    return RGB_mean


def get_image_pixel_std(img_dir, img_mean):
    R_squared_mean = 0
    G_squared_mean = 0
    B_squared_mean = 0
    count = 0
    image_mean = np.array(img_mean)
    # 循环读取所有图片
    for img_path in glob.glob(img_dir + '/*.jpg'):
        # img_path = os.path.join(img_dir, img_name)
        if not os.path.isdir(img_path):
            # image = cv2.imread(img_path)    # 读取图片
            image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image =image/255.0
            # image = cv2.resize(image, (img_size, img_size))      # <class 'numpy.ndarray'>
            image = image - image_mean    # 零均值
            # 求单张图片的方差
            R_squared_mean += np.mean(np.square(image[:, :, 0]).flatten())
            G_squared_mean += np.mean(np.square(image[:, :, 1]).flatten())
            B_squared_mean += np.mean(np.square(image[:, :, 2]).flatten())
            count += 1
    R_std = math.sqrt(R_squared_mean / count)
    G_std = math.sqrt(G_squared_mean / count)
    B_std = math.sqrt(B_squared_mean / count)
    print('R_std:{}, G_std:{}, B_std:{}'.format(R_std, G_std, B_std))
    RGB_std = [R_std, G_std, B_std]
    return RGB_std



class DataGenerator(Dataset):
    def __init__(self, label_dir, img_dir, mean , std, transform=None, target_transform=None):
        
        # with open('test.npy', 'rb') as f:
        #     self.img_labels= np.load(f)
        # self.label_dir = label_dir
        # self.img_dir = img_dir
        self.mean=mean
        self.std=std
        self.img_files = glob.glob(img_dir + '/*.jpg')
        self.label_files= []
        for img_path in self.img_files:
             self.label_files.append(os.path.join(label_dir,os.path.basename(img_path).split('.')[0]+'.npy')) 
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # image = cv2.imread(img_path)
        image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = cv2.imread(self.label_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        with open(label_path, 'rb') as f:
            label = np.load(f)
        
        if self.transform is not None:
            aug = self.transform(image=image, mask=label)
            image = Image.fromarray(aug['image'])
            label = aug['mask']
        
        if self.transform is None:
            image = Image.fromarray(image)
        
        p = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        image = p(image)
        label = torch.from_numpy(label).long()
        
        return image, label
    
        # image = Image.open(img_path)
        # # m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        # p = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=m, std=s),
        # ])
        # image = p(image)
        # # image = np.array(image)
        # # image = np.transpose(image, axes = [2,0,1])
        # with open(label_path, 'rb') as f:
        #     label = np.load(f)
        #     # ohlabel= torch.load(f)
        # # label = read_image(img_path)
        # # label = np.array(image)
        # ohlabel = torch.cuda.LongTensor(label)
        # # ohlabel = nn.functional.one_hot(label,22)
        # # ohlabel = ohlabel.permute(2,0,1)
        # # ohlabel = ohlabel.cpu().numpy()
        # # ohlabel = np.transpose(ohlabel, axes = [2,0,1])
        
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     ohlabel = self.target_transform(ohlabel)
        # return image, ohlabel
    
H = A.Compose([ A.HorizontalFlip()
                     ])
V = A.Compose([ A.VerticalFlip()
                     ])
HV = A.Compose([ A.HorizontalFlip(), A.VerticalFlip()
                     ])
Rotate90 = A.Compose([ 
                     A.RandomRotate90(p=1)
                     ])
HVRotate90 = A.Compose([ A.HorizontalFlip(), A.VerticalFlip(), 
                     A.RandomRotate90(p=1)
                     ])

# mean = get_image_pixel_mean(train_img_dir)
# std = get_image_pixel_std(train_img_dir, mean)
# mean = [0.41,0.483,0.341]
# std = [0.259,0.199,0.211]




    # torch.from_numpy(data).float()
    
class constants():
    def __init__(self):
        self.input_shape = (512,512,3)
        self.num_classes = 22
constants = constants()


def color_palette(input_theme, number_of_stops):
    # https://colorbrewer2.org/?type=sequential&scheme=Oranges&n=4#type=sequential&scheme=OrRd&n=8
    # single hue: Blues, Greens, Oranges, Purples, Reds
    # multi hue: BuGn, ...           Paired
    color_percentage = sns.color_palette(input_theme, number_of_stops)
    color_rgb = []
    for color in color_percentage:
        color_rgb.append( (int(round(list(color)[0]*255,0)), int(round(list(color)[1]*255,0)), int(round(list(color)[2]*255, 0))))
    return color_rgb
color_list = color_palette('hls', 22)

# input_gt = input_gt.cpu().numpy()

def to_color (color_list,input_gt):
    seg_img = np.zeros((constants.input_shape[0],constants.input_shape[1],3))
    seg_img = seg_img.astype('uint8')
    #        seg_img = np.zeros((320,320,3))
    for c in range(constants.num_classes):
        seg_img[:,:,0] += ((input_gt[:,: ] == c )*( color_list[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((input_gt[:,: ] == c )*( color_list[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((input_gt[:,: ] == c )*( color_list[c][2] )).astype('uint8')
    return seg_img



if __name__ == '__main__':
    # mean = get_image_pixel_mean(train_img_dir)
    # std = get_image_pixel_std(train_img_dir, mean)
    mean = [0.41,0.483,0.341]
    std = [0.259,0.199,0.211]

    
    train_data1 = DataGenerator(train_label_dir,train_img_dir, mean, std,)
    # train_data2 = DataGenerator(train_label_dir,train_img_dir, mean, std,H)
    # train_data3 = DataGenerator(train_label_dir,train_img_dir, mean, std,V)
    # train_data4 = DataGenerator(train_label_dir,train_img_dir, mean, std,HV)
    # train_data5 = DataGenerator(train_label_dir,train_img_dir, mean, std,Rotate90)
    # train_data6 = DataGenerator(train_label_dir,train_img_dir, mean, std,HVRotate90)

    # train_data = torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4, train_data5, train_data6])

    # test_data = DataGenerator(test_label_dir, test_img_dir)
    # train_data = DataGenerator(train_label_dir,train_img_dir,mean,std)
    test_data = DataGenerator(test_label_dir, test_img_dir,mean,std)
    
    train_dataloader = DataLoader(train_data1, batch_size= 4, shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size = 4, shuffle =True)
    train_features, train_labels = next(iter(train_dataloader))
    test_features, test_labels = next(iter(test_dataloader))
    
    
    
# ####################计算各个类别权重  
    category_simp_list=['房','车','船','林','路','水','园','草','污','构筑物','裸','人工堆叠','未分类']
    category_dict = dict(zip(category_simp_list,range(len(category_simp_list))))
    
    flag_list=[0]*13
    for ki,vi in category_dict.items():
        print(ki)
        for data in train_dataloader:
            for n in range(random.randint(1,data[1].size()[0]-1)):  #data[1]即为train_labels
                for h in range(512):
                    for w in range(512):
                        if data[1][n][h][w] == vi:
                            flag_list[vi]+=1
        
    # flag_list.sort()
    flag_med = sorted(flag_list)[6]
    weight_list = [flag_med/i for i in flag_list]########应对计算出的权重作适当调整，因为有的类别会是其他类别的几十倍，会导致训练精度下降
# weight_list = []
