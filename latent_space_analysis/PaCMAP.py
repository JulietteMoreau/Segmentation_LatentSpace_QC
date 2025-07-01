#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:31:11 2023

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

# numpy and pandas
import numpy as np

# Common python packages
import datetime
import os
import sys
import time

#monai stuff
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose
from monai.data import CacheDataset
import time
import numpy as np
import pandas as pd
import random as rd

import pacmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import os

from PIL import Image
import nibabel as nib



from encoder_UNet import UNet

xform = Compose([LoadImageD(('cerveau')),
    EnsureChannelFirstD(('cerveau')),
    ToTensorD(('cerveau'))])

bs = 1

data_dir = '/path/to/train/data/folder/'
test_dir = '/path/to/test/data/folder/'
val_dir = '/path/to/validation/data/folder/'

test_images = sorted(glob.glob(data_dir + "*.jpg")) + sorted(glob.glob(test_dir + "*.jpg"))+ sorted(glob.glob(val_dir + "*.jpg")) 

test_files=[]
for im in range(len(test_images)):
    test_files.append({"cerveau": test_images[im]})

test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

for i, batch in enumerate(test_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

encoder = UNet()

state_dict = torch.load("/path/to/checkpoint/weights/checkpoint.pth", map_location=lambda storage, loc: storage)

with torch.no_grad():
    encoder.inc.double_conv[0].weight.copy_(state_dict['inc.double_conv.0.weight'])
    encoder.inc.double_conv[1].weight.copy_(state_dict['inc.double_conv.1.weight'])
    encoder.inc.double_conv[3].weight.copy_(state_dict['inc.double_conv.3.weight'])
    encoder.inc.double_conv[4].weight.copy_(state_dict['inc.double_conv.4.weight'])
    encoder.down1.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.0.weight'])
    encoder.down1.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.1.weight'])
    encoder.down1.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.3.weight'])
    encoder.down1.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.4.weight'])
    encoder.down2.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.0.weight'])
    encoder.down2.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.1.weight'])
    encoder.down2.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.3.weight'])
    encoder.down2.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.4.weight'])
    encoder.down3.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.0.weight'])
    encoder.down3.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.1.weight'])
    encoder.down3.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.3.weight'])
    encoder.down3.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.4.weight'])
    encoder.down4.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.0.weight'])
    encoder.down4.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.1.weight'])
    encoder.down4.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.3.weight'])
    encoder.down4.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.4.weight'])

encoder.cuda()


data = []
label = []
coupe=[]
groupe = []
patient = []

for i, batch in enumerate(test_loader):
    

    torch.cuda.empty_cache()
    real_CT = batch["cerveau"].type(Tensor)
    image_latent = encoder(real_CT)

    image = Tensor.cpu(image_latent).detach().numpy()

    descripteurs = []
    for c in range(512):
        descripteurs.append(np.mean(image[0,c,:,:]))
        
    down4_BN2 = state_dict['down4.maxpool_conv.1.double_conv.4.weight']
    down4_BN2 = down4_BN2.cpu()
    down4_BN2 = down4_BN2.detach().numpy()
    down4_BN2 = list(down4_BN2)
    
    sorted_indices = sorted(range(len(down4_BN2)), key=lambda j: down4_BN2[j], reverse=True)   
    sorted_indices = sorted_indices[:len(sorted_indices)]
    
    kept_descripteurs = []
    for d in sorted_indices:
        kept_descripteurs.append(descripteurs[d])
    
    data.append(kept_descripteurs)
    if test_files[i]['cerveau'].split('/')[-1][0]=='p':
        label.append(1)
    else:
        label.append(2)
    coupe.append((test_files[i]['cerveau'].split('/')[-1]))
    patient.append((test_files[i]['cerveau'].split('/')[-1][0:5]))
    
    if test_files[i]['cerveau'].split('/')[-1] in os.listdir('/path/to/train/data/folder/'):
        groupe.append('train')
    elif test_files[i]['cerveau'].split('/')[-1] in os.listdir('/path/to/validation/data/folder/'):
        groupe.append('validation')
    else:
        groupe.append('test')
    
        

feat_cols = ['pixel'+str(i) for i in range(1,len(data[0])+1)]

df = pd.DataFrame(data, columns=feat_cols)

df['y'] = label
df['label'] = df['y'].apply(lambda i: str(i))
df['coupe'] = coupe
df['patient']=patient

data_pacmap = df[feat_cols].values

N = 60 # set to 10 for MRI data, 60 for CT data
time_start = time.time()
pacmap_res = pacmap.PaCMAP(n_components=2, n_neighbors=N, MN_ratio=0.5, FP_ratio=2) 
pacmap_results = pacmap_res.fit_transform(data_pacmap)

print('PACMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['pacmap-2d-one'] = pacmap_results[:,0]
df['pacmap-2d-two'] = pacmap_results[:,1]
df['groupe'] = groupe

df.to_csv('/path/to/output/file/latent_space.csv')

