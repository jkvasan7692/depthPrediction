# sys
import h5py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# torch
import torch
from torchvision import datasets, transforms


def load_data(_path, _ftype, coords, joints, cycles=3):

    # data path
    path_to_depth = './nyu_depth_v2_labeled.mat'

    # read mat file
    f = h5py.File(path_to_depth)

    # file_feature = os.path.join(_path, 'features2D' + _ftype + '.h5')
    # ff = h5py.File(file_feature, 'r')
    ff = f['images']
    # file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    # fl = h5py.File(file_label, 'r')
    fl = f['labels']

    data_list = []
    num_samples = len(ff.keys())
    num_frames = len(ff[list(ff.keys())[0]])
    print(num_samples , num_frames)
    time_steps = 0
    labels = np.empty(num_samples*num_frames)
    for si in range(num_samples):
        ff_group_key = list(ff.keys())[si]
        #print(si, ff_group_key)
        data_list.append(list(ff[ff_group_key]))  # Get the data
        time_steps_curr = np.shape(ff[ff_group_key])[1]
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels[si*num_frames:(si+1)*num_frames] = fl[list(fl.keys())[si]][()]

    data = np.empty((num_samples*num_frames, time_steps*cycles, joints*coords))
    print(np.shape(data))
    for si in range(num_samples):
        #data_list_curr = np.zeros((time_steps, np.size(data_list[si],1)))
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / np.shape(ff[ff_group_key])[1])), 1))
        print(np.shape(data_list_curr))
        #data_list_curr[0:k.shape[0],:] = k
        for ri in range(num_frames):
            for ci in range(cycles):
                data[si+ri, time_steps * ci:data_list_curr.shape[1], :] = data_list_curr[ri, 0:time_steps]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1)
    return data, labels, data_train, labels_train, data_test, labels_test


def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, label, joints, coords, num_classes):
        # data: N C T J
        self.data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords, 1))
        self.data = np.moveaxis(self.data, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.N, self.C, self.T, self.J, self.M = self.data.shape

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
