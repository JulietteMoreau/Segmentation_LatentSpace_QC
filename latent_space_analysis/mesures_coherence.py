#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:46:42 2025

@author: moreau
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from scipy.stats import gamma, kstest
import scipy


################################################################################################################ CT


# Load data
df = pd.read_csv("/path/to/projection/espace_latent_CT.csv")

X = df[["pca_branche_1", "pca_branche_2"]].values
n_values = df["log_aire"].values
sample_names = df["coupes"].values

k = 6  # Number of neighbors
alpha = 1  # Weighting exponent (higher values give more importance to closer points)
epsilon = 1e-6  # Small value to prevent division by zero

# Compute k+1 neighbors (including the point itself)
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(X)
distances, indices = nbrs.kneighbors(X)

# Compute Weighted MAD using inverse distance weighting
MAD_dict_weighted = {}

for i in range(len(n_values)):
    neighbor_idxs = indices[i][1:]  # Exclude self
    neighbor_distances = distances[i][1:]  # Exclude self
    neighbor_values = n_values[neighbor_idxs]

    # Compute inverse distance weights (avoid division by zero)
    weights = 1 / (neighbor_distances**alpha + epsilon)
    weights /= np.sum(weights)  # Normalize weights

    # Compute weighted MAD
    weighted_mad = np.sum(weights * np.abs(n_values[i] - neighbor_values))

    MAD_dict_weighted[sample_names[i]] = weighted_mad

 
# plot incoherence distribution
plt.figure(figsize=(16,10))
plt.hist(list(MAD_dict_weighted.values()), bins=50, color='indianred')
plt.xlabel('incoherence',fontsize=30)
plt.ylabel('frequency', fontsize=30)
plt.xlim([-0.1, 0.90])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

# normalized version
values = list(MAD_dict_weighted.values())
counts, bins = np.histogram(values, bins=50, range=(0,1))
normalized_counts = counts / counts.sum()
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(16,10))
plt.bar(bin_centers, normalized_counts, width=(bins[1]-bins[0]), color='indianred', align='center')
plt.xlabel('incoherence', fontsize=30)
plt.ylabel('normalized frequency', fontsize=30)
plt.xlim([-0.1, 0.90])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# compute threshold
data = list(MAD_dict_weighted.values())
incoherences = np.array(data)

shape, loc, scale = gamma.fit(incoherences) # gamma distribution
x = np.linspace(0, max(incoherences)*1.2, 1000)
pdf = gamma.pdf(x, a=shape, scale=scale)

mode = (shape - 1) * scale if shape > 1 else 0
peak_val = gamma.pdf(mode, a=shape, scale=scale)

half_max = peak_val / 2
indices_above_half = np.where(pdf >= half_max)[0]
fwhm_width = x[indices_above_half[-1]] - x[indices_above_half[0]]

lambda_factor = 1.5  # possible to tune this value
threshold = mode + lambda_factor * fwhm_width


# plot gamma fit
x = np.linspace(min(data), max(data), 100)
plt.hist(data, bins=30, alpha=0.6, density=True, color='g')
plt.plot(x, gamma.pdf(x, *[shape, loc, scale]), 'r-', label='Gamma fit')
plt.legend()
plt.xlim([-0.1, 0.90])
plt.show()

D, p_value = kstest(data, 'gamma', args=[shape, loc, scale])
print(f"Résultat du test K-S : D = {D:.4f}, p-value = {p_value:.4f}")
print("incoherence threshold", threshold)

count=0
CT_incoherent = []
for i,j in MAD_dict_weighted.items():
    if j>threshold:
        count+=1
        CT_incoherent.append(i)
        
print("number of incoherent slices", count)


liste_coupes = []
compte_pixel_moins_8=0
for i in CT_incoherent:
    im = Image.open('/path/to/imagedata/'+i)
    ref = Image.open('/path/to/mask/data/'+i[:-3]+'png')
    array = np.array(ref)
    liste_coupes.append(i)
    if np.count_nonzero(array[:,:,0])<=8:
        compte_pixel_moins_8+=1
    
    
print("number of slices with a smell lesion", compte_pixel_moins_8)


df = pd.read_excel('/path/to/features/2D_features_CT.xlsx')

aire = []
fisher = []
incoherence = []

# for c in liste_coupes:
for c in list(MAD_dict_weighted.keys()):
    aire.append(df.loc[df['slice'] == c[:-4], 'area'].iloc[0])
    fisher.append(df.loc[df['slice'] == c[:-4], 'fisher'].iloc[0])
    incoherence.append(MAD_dict_weighted[c])

plt.figure(figsize=(16,10))
plt.plot(incoherence, aire)
plt.xlabel('incoherence')
plt.ylabel('area')
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(incoherence, aire)
print("correlation area-incoherence", r_value**2)


plt.figure(figsize=(16,10))
plt.plot(incoherence, fisher)
plt.xlabel('incoherence')
plt.ylabel('fisher')
plt.show()


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(incoherence, fisher)
print("correlation fisher-incoherence", r_value**2)


print()



################################################################################################################ IRM


df = pd.read_csv("/path/to/projection/espace_latent_IRM.csv")

X = df[["pca_branche_1", "pca_branche_2"]].values
n_values = df["log_aire"].values
sample_names = df["coupes"].values

k = 6  # Number of neighbors
alpha = 1  # Weighting exponent (higher values give more importance to closer points)
epsilon = 1e-6  # Small value to prevent division by zero

# Compute k+1 neighbors (including the point itself)
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(X)
distances, indices = nbrs.kneighbors(X)

# Compute Weighted MAD using inverse distance weighting
MAD_dict_weighted = {}

for i in range(len(n_values)):
    neighbor_idxs = indices[i][1:]  # Exclude self
    neighbor_distances = distances[i][1:]  # Exclude self
    neighbor_values = n_values[neighbor_idxs]

    # Compute inverse distance weights (avoid division by zero)
    weights = 1 / (neighbor_distances**alpha + epsilon)
    weights /= np.sum(weights)  # Normalize weights

    # Compute weighted MAD
    weighted_mad = np.sum(weights * np.abs(n_values[i] - neighbor_values))

    MAD_dict_weighted[sample_names[i]] = weighted_mad

# plot incoherence
plt.figure(figsize=(16,10))
plt.hist(list(MAD_dict_weighted.values()), bins=50, color='indianred')
plt.xlabel('incoherence',fontsize=30)
plt.ylabel('frequency', fontsize=30)
plt.xlim([-0.1, 0.90])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

# normalized version
values = list(MAD_dict_weighted.values())
counts, bins = np.histogram(values, bins=50, range=(0,1))
normalized_counts = counts / counts.sum()
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(16,10))
plt.bar(bin_centers, normalized_counts, width=(bins[1]-bins[0]), color='indianred', align='center')
plt.xlabel('incoherence', fontsize=30)
plt.ylabel('normalized frequency', fontsize=30)
plt.xlim([-0.1, 0.90])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

# compute threshold
data = list(MAD_dict_weighted.values())
incoherences = np.array(data)

shape, loc, scale = gamma.fit(incoherences) # gamma distribution
x = np.linspace(0, max(incoherences)*1.2, 1000)
pdf = gamma.pdf(x, a=shape, scale=scale)

mode = (shape - 1) * scale if shape > 1 else 0
peak_val = gamma.pdf(mode, a=shape, scale=scale)

half_max = peak_val / 2
indices_above_half = np.where(pdf >= half_max)[0]
fwhm_width = x[indices_above_half[-1]] - x[indices_above_half[0]]

lambda_factor = 1.5  # possible to tune this value
threshold = mode + lambda_factor * fwhm_width



# plot gamma fit
x = np.linspace(min(data), max(data), 100)
plt.hist(data, bins=30, alpha=0.6, density=True, color='g')
plt.plot(x, gamma.pdf(x, *[shape, loc, scale]), 'r-', label='Gamma fit')
plt.legend()
plt.xlim([-0.1, 0.90])
plt.show()

D, p_value = kstest(data, 'gamma', args=[shape, loc, scale])
print(f"Résultat du test K-S : D = {D:.4f}, p-value = {p_value:.4f}")
print("incoherence threshold", threshold)

count=0
CT_incoherent = []
for i,j in MAD_dict_weighted.items():
    if j>threshold:
        count+=1
        CT_incoherent.append(i)
        
print("number of incoherent slices", count)


liste_coupes = []
compte_pixel_moins_8=0
for i in CT_incoherent:
    im = Image.open('/path/to/imagedata/'+i)
    ref = Image.open('/path/to/mask/data/'+i[:-3]+'png')
    array = np.array(ref)
    liste_coupes.append(i)
    if np.count_nonzero(array[:,:,0])<=8:
        compte_pixel_moins_8+=1
    
    
print("number of slices with a small lesion", compte_pixel_moins_8)

df = pd.read_excel('/path/to/features/2D_features_IRM.xlsx')

aire = []
fisher = []
incoherence = []

# for c in liste_coupes:
for c in list(MAD_dict_weighted.keys()):
    aire.append(df.loc[df['slice'] == c[:-4], 'area'].iloc[0])
    fisher.append(df.loc[df['slice'] == c[:-4], 'fisher'].iloc[0])
    incoherence.append(MAD_dict_weighted[c])

plt.figure(figsize=(16,10))
plt.plot(incoherence, aire)
plt.xlabel('incoherence')
plt.ylabel('area')
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(incoherence, aire)
print("correlation area-incoherence", r_value**2)


plt.figure(figsize=(16,10))
plt.plot(incoherence, fisher)
plt.xlabel('incoherence')
plt.ylabel('fisher')
plt.show()


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(incoherence, fisher)
print("correlation fisher-incoherence", r_value**2)