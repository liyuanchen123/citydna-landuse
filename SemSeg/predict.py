# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:21:31 2022

@author: GREEN&LYC
"""
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

class constants():
    def __init__(self):
        self.input_shape = (1024,1024,3)
        self.num_classes = 13
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


def predict_image_pixel(model, image, mean=[0.41,0.483,0.341], std=[0.259,0.199,0.211]):
    # mean = [0.41,0.483,0.341]
    # std = [0.259,0.199,0.211]
    model.eval()
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    # mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        
        output = model(image)
        # acc = pixel_accuracy(output, mask)
        output = torch.argmax(output, dim=1)
        # masked = masked.cpu().squeeze(0)
    return output

device = "cuda" if torch.cuda.is_available() else "cpu"
model= torch.load(r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/Code/landuse_diff/project/SemSeg/u-resnet50wCE.pt')
predict_image = Image.open(os.path.join(r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/input/hw1_cut_tiles/435384_222568.jpg'))
# predict_image = Image.open(os.path.join(r'/z/00INTERN/LiYuanchen/landuse_diff/data/input/hw1_cut_tiles/435380_222562.jpg'))
predict_image = np.array(predict_image)
pimage = predict_image_pixel(model,predict_image)
pimage=pimage.cpu().numpy()
pimage = pimage.squeeze()
color_list = color_palette('hls', 13)
result_image = to_color(color_list, pimage)
result_image = Image.fromarray(result_image)
result_image.save(os.path.join(r'Z:\A工作\landuse\landuse_diff-liyuanchen\SemSeg\pred_uresnet50_wCE.jpg'))
