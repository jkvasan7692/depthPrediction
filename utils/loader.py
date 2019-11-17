# sys
import h5py
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# torch
import torch
from torchvision import datasets, transforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

#
# def load_data(_path):
#     """
#     Requirement- Takes the file path, builds the train, test dataset images, depth map labels and returns them
#     :param _path:
#     :return:
#     data - Entire set of images
#     label - Entire set of depth map ground truth
#     data_train - training set of images
#     data_test - Testing set of images
#     labels_train - Depth map ground truth for training
#     labels_test - Depth maps ground truth for testing
#     """
#     # data path
#     path_to_depth = './nyu_depth_v2_labeled.mat'
#
#     # read mat file
#     f = h5py.File(path_to_depth)
#
#     # file_feature = os.path.join(_path, 'features2D' + _ftype + '.h5')
#     # ff = h5py.File(file_feature, 'r')
#     fd = f['images']
#     # file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
#     # fl = h5py.File(file_label, 'r')
#     # fl = f['depths']
#     # print(ff.shape)
#     num_samples, channel, width, height = fd.shape
#     # fd = np.array(fd)
#     # print(fd[0])
#     data = np.empty([num_samples, height, width, channel])
#     for i in range(num_samples):
#         img = fd[i].copy()
#         img_ = np.empty([height, width, channel])
#         img_[:,:,0] = img[0,:,:].T
#         img_[:,:,1] = img[1,:,:].T
#         img_[:,:,2] = img[2,:,:].T
#         # img__ = img_.astype('float32')/255.0
#         data[i,:,:,:] = img_.copy()
#
#     del img_
#     del img
#     del fd
#
#     depth_data = np.empty([num_samples, height, width])
#     for i in range(num_samples):
#         # depth_img = fl[0].copy()
#         depth_data[i,:,:] = f['depths'][i].copy().T
#
#     del f
#
#     # print(img__)
#     # plt.imshow(depth_data[0,:,:]/4)
#     # cv2.imshow('Lane Detection', depth_data[0,:,:]/4)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # data =
#     # for i in range(num_samples):
#
#     #
#     # data_list = []
#     # num_samples = len(ff.keys())
#     # num_frames = len(ff[list(ff.keys())[0]])
#     # print(num_samples , num_frames)
#     # time_steps = 0
#     # labels = np.empty(num_samples*num_frames)
#     # for si in range(num_samples):
#     #     ff_group_key = list(ff.keys())[si]
#     #     #print(si, ff_group_key)
#     #     data_list.append(list(ff[ff_group_key]))  # Get the data
#     #     time_steps_curr = np.shape(ff[ff_group_key])[1]
#     #     if time_steps_curr > time_steps:
#     #         time_steps = time_steps_curr
#     #     labels[si*num_frames:(si+1)*num_frames] = fl[list(fl.keys())[si]][()]
#     #
#     # data = np.empty((num_samples*num_frames, time_steps*cycles, joints*coords))
#     # print(np.shape(data))
#     # for si in range(num_samples):
#     #     #data_list_curr = np.zeros((time_steps, np.size(data_list[si],1)))
#     #     data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / np.shape(ff[ff_group_key])[1])), 1))
#     #     print(np.shape(data_list_curr))
#     #     #data_list_curr[0:k.shape[0],:] = k
#     #     for ri in range(num_frames):
#     #         for ci in range(cycles):
#     #             data[si+ri, time_steps * ci:data_list_curr.shape[1], :] = data_list_curr[ri, 0:time_steps]
#     data_train, data_test, labels_train, labels_test = TrainTestSplit(data, depth_data, 0.1)
#     labels = depth_data.copy()
#     return  data, labels, data_train, labels_train, data_test, labels_test
#
# def TrainTestSplit(data, label, ratio):
#     num_samples = data.shape[0]
#     num_train_samples = int(num_samples * (1 - ratio))
#     train_choices = np.random.choice(range(num_samples), num_train_samples, replace=False)
#     test_choices = np.setdiff1d(np.arange(num_samples), train_choices)
#
#     data_train = data[train_choices, :,:,:]
#     labels_train = label[train_choices, :,:]
#     data_test = data[test_choices, :,:,:]
#     labels_test = label[test_choices, :,:]
#
#     return data_train, data_test, labels_train, labels_test

# All three below functions not being used.
# def scale(_data):
#     data_scaled = _data.astype('float32')
#     data_max = np.max(data_scaled)
#     data_min = np.min(data_scaled)
#     data_scaled = (_data-data_min)/(data_max-data_min)
#     return data_scaled, data_max, data_min
#
#
# # descale generated data
# def descale(data, data_max, data_min):
#     data_descaled = data*(data_max-data_min)+data_min
#     return data_descaled


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

# Requirement here: self.data and self.label is important to us that should contain the images and depth map

class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, train = True):
        # data: N C T J
        np.random.seed(0)

        self.data, self.label = self.read_data(train)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

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
