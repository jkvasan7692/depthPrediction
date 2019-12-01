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
from nyuv2 import *
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
        if self.dataset == "labeled":
            label_tensor = torch.from_numpy(self.label[index])
            #print("Size of label data", label.shape)

            data_tensor = torch.from_numpy(self.data[index])
            #print("Size of input data", data_tensor.shape)
            data = self.transform1(data_tensor)
            #print("Shape after PIL image conversion", data.shape)
            data = self.transform2(data)
            #print("Shape after Center Crop and Resize image conversion", data.shape)
        else:
            label_pil = self.label[index]
            image_pil = self.data[index]
            data = self.transform2(image_pil)
            label_tensor = transforms.ToTensor(label_pil)
        print("Data shape", data.shape)
        print("Label shape", label_tensor.shape)
        return data, label_tensor

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
        data_numpy = np.array(data).astype('float32')/255.0
        img = np.zeros((3, data_numpy.shape[2], data_numpy.shape[1]), dtype='float32')
        img[0,:,:] = data_numpy[0,:,:].T
        img[1,:,:] = data_numpy[1,:,:].T
        img[2,:,:] = data_numpy[2,:,:].T

        label = np.array(f['depths'])[idxs]
        label_processed = label/10
        label = np.expand_dims(label_processed, axis=0)
        label_ = np.zeros((1, label_processed.shape[2], label_processed.shape[1]), dtype='float32')
        label_[0,:,:] = label_processed[0,:,:].T
        return data, img_

    def read_data_raw(self, train):
        """
        Reads the raw data processes and builds the dataset for training
        :param train:
        :return:
        """
        data = []
        label = []
        path_to_depth = Path('/home/jkv92/data')
        for elems in path_to_depth.glob('*.zip'):
            raw_archive = RawDatasetArchive(elems)
            print("Length is ", len(raw_archive), elems)

            for ind in range(0, len(raw_archive), 15):
                frame = raw_archive[ind]
                depth_path = path_to_depth / frame[0]
                image_path = path_to_depth / frame[1]

                if not (depth_path.exists() and image_path.exists()):
                    raw_archive.extract_frame(frame)
                #print(depth_path, image_path)

                if depth_path.stat().st_size > 0 and image_path.stat().st_size > 0:
                    color = load_color_image(image_path)
                    depth = load_depth_image(depth_path)
                    #print(color)
                    #print(depth)
                    data.append(color)
                    label.append(depth)
        return data, label
