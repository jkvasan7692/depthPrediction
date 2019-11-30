# sys
import h5py
import os
import sys
import numpy as np
# import tensorflow as tf
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import skimage.io as io

from pathlib import Path
from nyuv2_python_toolbox.nyuv2 import *
# torch
import torch
from torchvision import datasets, transforms
# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# except:
#     pass
# import cv2

class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, train = "train", dataset="labeled"):
        # data: N C T J
        np.random.seed(0)

        if dataset == "labeled:":
            self.data, self.label = self.read_data(train)
        else:
            self.data, self.label = self.read_data_raw(train)

        self.transform1 = transforms.Compose([transforms.ToPILImage()])
        self.transform2 = transforms.Compose([transforms.Resize(size=(240, 320), interpolation=1),
                                              transforms.CenterCrop(size=(228, 304)), transforms.ToTensor()])

        self.dataset = dataset

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
        #print("Size of label data", label.shape)

        data_tensor = torch.from_numpy(img_)
        #print("Size of input data", data_tensor.shape)
        data = self.transform1(data_tensor)
        #print("Shape after PIL image conversion", data.shape)
        data = self.transform2(data)
        #print("Shape after Center Crop and Resize image conversion", data.shape)
        return data, label_

    def read_data(self, train):
        # data path
        path_to_depth = '../data/nyu_depth_v2_labeled.mat'

        # read mat file
        f = h5py.File(path_to_depth)
        num_samples = f['images'].shape[0]
        train_idx = int(num_samples*0.8)
        test_idx = int(num_samples*0.9)
        if train == "train":
            idxs = np.arange(train_idx)
        elif train == "eval":
            idxs = np.arange(train_idx + 1, test_idx)
        else:
            idxs = np.arange(test_idx+1, num_samples - 1)
        data = np.array(f['images'])[idxs]
        label = np.array(f['depths'])[idxs]
        return data, label

    def read_data_raw(self, train):
        """
        Reads the raw data processes and builds the dataset for training
        :param train:
        :return:
        """
        data = []
        label = []
        path_to_depth = Path('/media/kirthi/Seagate Backup Plus Drive/MLProjectDataset')
        for elems in path_to_depth.glob('*.zip'):
            raw_archive = RawDatasetArchive(elems)

            for ind in range(len(raw_archive)):
                frame = raw_archive[ind]
                depth_path = Path('.') / frame[0]
                image_path = Path('.') / frame[1]

                if not (depth_path.exists() and color_path.exists()):
                    raw_archive.extract_frame(frame)

                color = load_color_image(image_path)
                depth = load_depth_image(depth_path)

                print("Shape of the color image is", color.shape)
                print("Shape of the depth image is", depth.shape)

                data.append(color)
                label.append(depth)

        data_arr = np.array(data)
        label_arr = np.array(label)
        print(data_arr.shape())
        print(label_arr.shape())
        return data_arr, label_arr