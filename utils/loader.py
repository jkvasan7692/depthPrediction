# sys
import h5py
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io as io
# torch
import torch
from torchvision import datasets, transforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, train = True):
        # data: N C T J
        np.random.seed(0)

        self.data, self.label = self.read_data(train)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(240,320), interpolation=1),
            transforms.CenterCrop(size=(228,304)),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index]).astype('float32')/255.0
        # print(np.amax(data_numpy))
        # asd
        label = self.label[index]/10
        img_ = np.zeros((3, data_numpy.shape[2], data_numpy.shape[1]), dtype='float32')
        img_[0,:,:] = data_numpy[0,:,:].T
        img_[1,:,:] = data_numpy[1,:,:].T
        img_[2,:,:] = data_numpy[2,:,:].T

        # print(data_numpy)
        # print(type(data_numpy))
        label = np.expand_dims(label, axis=0)
        label_ = np.zeros((1, label.shape[2], label.shape[1]), dtype='float32')
        label_[0,:,:] = label[0,:,:].T
        label_ = torch.from_numpy(label_)

        data_tensor = torch.from_numpy(img_)
        data = self.transform(data_tensor)
        return data, label_

    def read_data(self, train):
        # data path
        path_to_depth = './nyu_depth_v2_labeled.mat'

        # read mat file
        f = h5py.File(path_to_depth)
        num_samples = f['images'].shape[0]
        idx = int(num_samples*0.9)
        if train:
            idxs = np.arange(idx)
        else:
            idxs = np.arange(idx + 1, num_samples-1)
        data = np.array(f['images'])[idxs]
        label = np.array(f['depths'])[idxs]
        return data, label
