import argparse
import os
import sys
import numpy as np
#from utils import processor1, loader
from net import classifier1
import torch.nn as nn
#import torchvision
#from torchvision import transforms


import torch
#import torchlight
#import skimage.io as io


def unit_test_loss():
    """
    Function to unit test the reverse hubber loss class
    :return:
    """
    x = torch.randn(2, 2, requires_grad=True)
    target = torch.randn(2, 2, requires_grad=True)
    # print(x)
    # print(target)

    loss_fn = processor1.ReverseHubberLoss()
    output = loss_fn.forward(x, target)
    # print(output)
    # print(output.backward())
    return

def weight_data(m):
    """
    Unit test to apply a function to print weight data
    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d):
        print(m.weight.data)

def unit_test_model():
    """
    Function to unit test the model
    :return:
    """
    model = classifier1.DepthPredictionNet()
    print(model)
    x = torch.randn(64, 3, 304, 228)
    print(x.shape)
    y = model.forward(x)
    print(y.shape)
    return

# Loader class parameters
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')
ftype = ''

# Path for saved models
# model_path = os.path.join(base_path, 'model_classifier_stgcn/features2D'+ftype)

#%% - Parsing Arguments
parser = argparse.ArgumentParser(description='Depth prediction')
# Arguments for training the model - Default value
# Training - True
parser.add_argument('--train', type=bool, default=False, metavar='T',
                    help='train the model (default: True)')
# Saving the features - True. Currently disabled
parser.add_argument('--save-features', type=bool, default=True, metavar='SF',
                    help='save penultimate layer features (default: True)')

# Batch size - 8
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')

# Number of Workers - 4
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')

# Starting epoch for training - 0
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')

# Number of epochs - 500
parser.add_argument('--num_epoch', type=int, default=500, metavar='NE',
                    help='number of epochs to train (default: 500)')

# Optimizer - Adam
parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                    help='optimizer (default: SGD)')

# Learning rate - 0.1
parser.add_argument('--base-lr', type=float, default=0.1, metavar='L',
                    help='base learning rate (default: 0.1)')

# Modification of learning rate steps - [0.5, 0.75, 0.875]. Currently not adjusting the learning rate
parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')

# Nesterov - Dunno what is this, branched of the parent code
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')

# Momentum rate - 0.9
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')

# Weight-decay rate - 5*10^-4
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')

# Evaluation interval - 1
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')

# Logging interval - 100
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')

# Topk [1]. Currently disabled
parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                    help='top K accuracy to show (default: [1])')

# No-cuda - False
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# Print Log - True
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')

# save Log - True
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')

# Working directory - model_path. Currently disabled
# parser.add_argument('--work-dir', type=str, default=model_path, metavar='WD',
#                     help='path to save')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'

unit_test_model()
# pr = processor1.Processor(args, None, device)
# data, labels, data_train, labels_train, data_test, labels_test = \
#     loader.load_data(data_path)
# a = loader.TrainTestLoader(False)
# d,l = a.__getitem__(0)
# print(type(d))
# print(type(l))
# print(l.shape)
# print(d.shape)
# # print(np.amax(d))
# img__ = d.numpy()
# # print(np.amax(l))
# img_ = np.zeros((d.shape[1], d.shape[2], 3))
# img_[:,:,0] = d[0,:,:]
# img_[:,:,1] = d[1,:,:]
# img_[:,:,2] = d[2,:,:]
# img__ = img_.astype('float32')
# io.imshow(img__)
# io.show()
# # This part of the code should work after the above is resolved.
# # data_loader_train_test.append(torch.utils.data.DataLoader(
# #     dataset=loader.TrainTestLoader(data_train, labels_train, joints, coords, num_classes),
# #     batch_size=args.batch_size,
# #     shuffle=True,
# #     num_workers=args.num_worker * torchlight.ngpu(device),
# #     drop_last=True))
# # data_loader_train_test.append(torch.utils.data.DataLoader(
# #     dataset=loader.TrainTestLoader(data_test, labels_test, joints, coords, num_classes),
# #     batch_size=args.batch_size,
# #     shuffle=True,
# #     num_workers=args.num_worker * torchlight.ngpu(device),
# #     drop_last=True))
# # data_loader_train_test = dict(train=data_loader_train_test[0], test=data_loader_train_test[1])
