#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:25:22 2022

@author: moreau
"""

import matplotlib.pyplot as plt
import os
import glob

# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torchsummary and torchvision
from torchsummary import summary
from torchvision.utils import save_image

# matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.image as img

# numpy and pandas
import numpy as np
import pandas as pd

# Common python packages
import datetime
import os
import sys
import time

#monai stuff
from monai.transforms import RandSpatialCropSamplesD,SqueezeDimd, SplitChannelD,RandWeightedCropd,\
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose, CropForegroundd,\
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandAffineD, CopyItemsd, OneOf, RandCoarseDropoutd, RandFlipd
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################################################################################################################################load data

dossier = "/path/to/images/" 
dossier_pat = '/path/to/3D/images/' 
fold_val = 1 #one run for each of the 5 folds
outdir = "/output/directory/"


if not os.path.exists(outdir):
    os.makedirs(outdir)   

KEYS = ("cerveau", "GT")

train_dir = dossier + '/image/train/'
train_dir_label = dossier + '/ref/train/'
val_dir = dossier + '/image/validation/'
val_dir_label = dossier + '/ref/validation/'
test_dir = dossier + '/image/test/'
test_dir_label = dossier + '/ref/test/'

# ordered images and references name
images = sorted(glob.glob(train_dir + "*.jpg") + glob.glob(val_dir + "*.jpg") + glob.glob(test_dir + "*.jpg"))
labels = sorted(glob.glob(train_dir_label + "*.png") + glob.glob(val_dir_label + "*.png") + glob.glob(test_dir_label + "*.png"))

def split_into_five_parts(data):
    """
    split the dataset into five folds
    """
    n = len(data)
    base_size = n // 5
    extras = n % 5

    subsets = []
    start = 0
    for i in range(5):
        size = base_size + (1 if i < extras else 0)
        end = start + size
        subsets.append(data[start:end])
        start = end
    return subsets

def get_subset_and_others(data, k):
    """
    function to obtain the data separated in validation and test
    k is the number of the fold of the validation set, between 1 and 5
    """
    subsets = split_into_five_parts(data)
    subset_k = subsets[k - 1]
    others = [item for i, subset in enumerate(subsets) if i != (k - 1) for item in subset]
    return subset_k

# ordered patients names
patients = sorted(os.listdir(dossier_pat))

# split into train and validation based on patients names to have all slices of one patient in the same fold
val_pat = get_subset_and_others(patients, fold_val)

val_images, val_labels = [], []
train_images, train_labels = [], []

# split 2D images into the train and validation sets
for i in images:
    print(i[-17:-13])
    if i[-18:-13]+'.nii.gz' in val_pat:
        val_images.append(i)
    else:
        train_images.append(i)
  
# split 2D labels into the train and validation sets
for i in labels:
    if i[-18:-13]+'.nii.gz' in val_pat:
        val_labels.append(i)
    else:
        train_labels.append(i)

train_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images, train_labels)
]
val_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images, val_labels)
]

print("Number Train files: "+str(len(train_files)))
print("Number val files: "+str(len(val_files)))

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 12
train_ds = CacheDataset(data=train_files, transform=xform, num_workers=10)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=xform, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True)


for i, batch in enumerate(train_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
for i, batch in enumerate(val_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()



################################################################################################################paramètres d'apprentissage
# Parameters for Adam optimizer
lr = 0.0001 


# Number of epochs
num_epoch = 100
patience = 10


# ################################################################################################################generator Unet
from generator_UNet import UNet
# Summary of the generator
summary(UNet().cuda(), (1, 192, 192))

from train_output_constrained import train_net
generator = train_net(train_loader, train_ds, val_loader, outdir, patience=patience, num_epoch=num_epoch, lr=lr)
