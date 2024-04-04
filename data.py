# %%
import pandas as pd
import numpy as np
import os
import random

# %%
# 根据文件夹的名称确定标签
label = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    '11': 11, '12': 12, '13': 13,
    '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
    '21': 21, '22': 22, '23': 23, '24': 24
}


# %%
# 把图片路径和对应标签装到python列表里
def prepare_data(data_file: str):
    data_list = []
    for i in os.listdir(data_file):
        for j in os.listdir(data_file + '/' + i):
            img_path = data_file + '/' + i + '/' + j
            img_label = label[i]
            img_data = [img_path, img_label]
            data_list.append(img_data)
    return data_list


# %%
# 把含有图片路径和标签的列表存到csv文件中，读取数据的时候根据csv文件的指示读取
# data_file:指Hand_Posture_Hard_Stu
# csv_path:存到的csv文件路径，shuffle:是否打乱列表再存入
def data_to_csv(data_file: str, csv_path: str = './workers_ind/data.csv', Shuffle=False):
    data_list = prepare_data(data_file)
    if Shuffle:
        random.shuffle(data_list)
    data_csv = pd.DataFrame(data_list)
    data_csv.to_csv(csv_path, index=False)


# %%
# 把data.csv文件分为训练集，测试集，验证集
# train,test,valid:训练集，测试集，验证集的比例
# shuffle:是否打乱再分
def csv_split_train_test_valid(csv_path='./data.csv', train=0.6, test=0.2, valid=0.2, Shuffle=False):
    l = train + test + valid
    train, test, valid = train / l, test / l, valid / l
    data = pd.read_csv('./workers_ind/data.csv')
    data_num = len(data)
    train_num = int(data_num * train)
    test_num = int(data_num * test)
    valid_num = data_num - train_num - test_num
    if Shuffle:
        index = list(range(data_num))
        random.shuffle(index)
        data = data.iloc[index]
    train_data = data[:train_num]
    test_data = data[train_num:train_num + test_num]
    valid_data = data[train_num + test_num:]
    train_data.to_csv('./workers_ind/train.csv', index=False)
    test_data.to_csv('./workers_ind/test.csv', index=False)
    valid_data.to_csv('./workers_ind/valid.csv', index=False)


# %%
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image


# %%
# torch读取数据，继承Dataset,需要定义三个函数
# __init__：传入相应的数据，这里传入对应的csv文件路径和transforms.
# __getitem__ : index表示读第几个数据，该函数返回图像数据和标签(必须是tensor类型)
# __len__:返回数据的个数
class imgdataset(Dataset):
    def __init__(self, data, transforms=False):
        self.data = pd.read_csv(data)
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.data.iloc[index][0]
        label = int(self.data.iloc[index][1])
        img = Image.open(img)
        if self.transforms != False:
            img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.int64)
        return img, label

    def __len__(self):
        return len(self.data)


# %%
def imgloader(data, transforms=False, batch_size=64, shuffle=True):
    mydataset = imgdataset(data, transforms)
    return DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle)




# %%
from torchvision import transforms


# %%
# 训练时的数据增强
def train_transforms():
    return transforms.Compose([
        transforms.RandomInvert(p=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# %%
# 测试时的数据处理
def test_transforms():
    return transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
