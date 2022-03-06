# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:44:23 2021

@author: GREEN&LYC
"""
import json
import os
import shutil
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import seaborn as sns
# import skimage.io as io
from labelme import utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from collections import defaultdict


all_json_path = r'\\192.168.23.247\ctdna_common\00PROJECTS\Research and Development\RM_2020_04.卫星影像深度学习\satellite_image_panoptic_seg\data\landuse_0507'
train_img_path = r'\\192.168.23.247\ctdna_common\00INTERN\LiYuanchen\landuse_diff\data\train'
test_img_path = r'\\192.168.23.247\ctdna_common\00INTERN\LiYuanchen\landuse_diff\data\test'
train_json_path = r'\\192.168.23.247\ctdna_common\00INTERN\LiYuanchen\landuse_diff\data\train_json'
test_json_path = r'\\192.168.23.247\ctdna_common\00INTERN\LiYuanchen\landuse_diff\data\test_json'
train_label_path = r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train_label'
test_label_path = r'//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/test_label'

#%%

ls = [item.split(".")[0] for item in os.listdir(all_json_path) if item.endswith("json")]

for train_img in os.listdir(train_img_path):
    if train_img.split('.')[0] in ls:
        js_name = train_img.split('.')[0] + ".json"
        js_path = all_json_path + "\\" + js_name
        # print(js_path)
        shutil.copyfile(js_path,os.path.join(train_json_path,js_name))
        
for test_img in os.listdir(test_img_path):
    if test_img.split('.')[0] in ls:
        js_name = test_img.split('.')[0] + ".json"
        js_path = all_json_path + "\\" + js_name
        # print(js_path)
        shutil.copyfile(js_path,os.path.join(test_json_path,js_name))
#%%   
###################################labelme json文件转化为one_hot 并对不同类上色，可视化
            
category_list = ["房屋建筑","林草覆盖,混合林","铁路与道路,乡村道路","林草覆盖,灌木林","构筑物,硬化地表","林草覆盖,乔木林",\
                 "未分类","铁路与道路,公路","其他类别,汽车","种植土地,水田、旱地、菜地","铁路与道路,铁路","林草覆盖,草地",\
                 "其他类别,水面油污","水域","荒漠与裸露地,泥土地表","林草覆盖,稀疏林","其他类别,船舶与浮筒","荒漠与裸露地,砾石地表",\
                 "人工堆掘地,堆放物","荒漠与裸露地,砂质地表","种植土地,果园、茶园等","人工堆掘地,露天采掘场"]

category_simp_list=['房','车','船','林','路','水','园','草','污','构筑物','裸','人工堆叠','未分类']

'''          这里将原本labelme json文件中的22类简化为13类，修改保存新json文件
category_simp_dict = defaultdict(list)
for i in category_list:
    for j in category_simp_list:
        if j in i:
            category_simp_dict[j].append(i)
# for i in category_simp_dict:
#     if len(category_simp_dict[i])==1:
#         category_simp_dict[i] = category_simp_dict[i][0]

category_simp_dict['林'].pop(3)
category_simp_dict['水']=['水域']        
category_simp_dict['园']= ['种植土地,果园、茶园等',"种植土地,水田、旱地、菜地"]
category_simp_dict['草']=['林草覆盖,草地']

category_simp_dict['人工堆叠']= ['人工堆掘地,堆放物',"人工堆掘地,露天采掘场"]
category_simp_dict=dict(category_simp_dict)

#将原json中标签映射为简化版 
def mapsimp(x):
    for k,values in category_simp_dict.items():
        for v in values:
            if x == v:
                return k
            
for json_name in os.listdir(test_json_path):
    with open(os.path.join(test_json_path,json_name),'r',encoding = 'utf-8') as json_file:
              data = json.load(json_file)  
    for i in data['shapes']:
        i['label'] = list(map(mapsimp,[i['label']]))[0]
    with open(os.path.join(test_json_path,json_name),'w',encoding = 'utf-8') as output_file:
              data = json.dump(data,output_file) 
'''




category_dict = dict(zip(category_simp_list,range(len(category_simp_list))))

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
# train_ls = 
#%%
################转换训练集label
for json_name in os.listdir(train_json_path):
    with open(os.path.join(train_json_path,json_name),'r',encoding = 'utf-8') as json_file:
              data = json.load(json_file)    
    # img = Image.open('//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train/542376_448364.jpg')
    img = Image.open(os.path.join(train_img_path,json_name.split('.')[0] + '.jpg'))
    img = np.array(img)
    lab, lab_names = utils.shapes_to_label(img.shape, data['shapes'],category_dict)
    
    # lab = torch.cuda.LongTensor(lab)
    # ohlab = nn.functional.one_hot(lab,22)
    #保存label为数组，颜色仅作为可视化展示，最终是需要对应的one_hot编码
    with open(os.path.join(train_label_path,json_name.split('.')[0] + '.npy'), 'wb') as f:
        np.save(f, lab)
        # torch.save(ohlab, f)
    # result_img = to_color(color_list, lab)
    # result_img = Image.fromarray(result_img)
    # result_img.save(os.path.join(train_label_path,json_name.split('.')[0] + '.jpg'))

#%%
#####################转换测试集label
for json_name in os.listdir(test_json_path):
    data = json.load(open(os.path.join(test_json_path,json_name),encoding = 'utf-8'))    
    # img = Image.open('//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train/542376_448364.jpg')
    img = Image.open(os.path.join(test_img_path,json_name.split('.')[0] + '.jpg'))
    img = np.array(img)
    lab, lab_names = utils.shapes_to_label(img.shape, data['shapes'],category_dict)
    
    # lab = torch.cuda.LongTensor(lab)
    # ohlab = nn.functional.one_hot(lab,22)
    # ohlab = ohlab.cpu().numpy()
    #保存label为数组，颜色仅作为可视化展示，最终是需要对应的one_hot编码
    with open(os.path.join(test_label_path,json_name.split('.')[0] + '.npy'), 'wb') as f:
        np.save(f, lab)
        # torch.save(ohlab, f)
    # result_img = to_color(color_list, lab)
    # result_img = Image.fromarray(result_img)
    # result_img.save(os.path.join(test_label_path,json_name.split('.')[0] + '.jpg'))
#%%    测试转换的label
with open('//192.168.23.247/ctdna_common/00INTERN/LiYuanchen/landuse_diff/data/train_label/542472_448354.pt', 'rb') as f:
        lab = torch.load(f)
lab = lab.cpu().numpy()
# # lab = torch.from_numpy(lab)

# # lab = torch.LongTensor(lab)
# lab = torch.cuda.LongTensor(lab)
# ohlab = nn.functional.one_hot(lab,22)
