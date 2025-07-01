#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:51:17 2025

@author: moreau
"""

import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, to_rgb, LogNorm
import matplotlib.pyplot as plt
import colorsys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr

colors = ["#fce75f","#f7dc5b", "#f1d257", '#ecc753','#e4b74d','#dda747','#d29240', "#c87c38", "#bd6730", "#b04c27", '#a3321d', '#991c15', '#991c15']
# colors = ['tan','dodgerblue', 'hotpink']
original_colors = [to_rgb(color) for color in colors]
pastel_colors = []
for rgb in original_colors:
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s *= 1  # Decrease saturation
    l *= 1  # Increase lightness
    adjusted_rgb = colorsys.hls_to_rgb(h, l, s)
    pastel_colors.append(adjusted_rgb)
custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', pastel_colors, N=256)




# =============================================================================
# COMPARISON OF THE THREE MODELS LATENT SPACES
# REPLACE "pcamap" BY "pca" TO SEE THE OTHER PROJECTIONS 
# =============================================================================

################## CT

df = pd.read_csv("/path/to/projections/espace_latent_CT.csv")


model = smf.mixedlm("log_aire ~ pacmap_ref_1 + pacmap_ref_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_ref_1'] * np.array(df['pacmap_ref_1']) + result.params['pacmap_ref_2'] * np.array(df['pacmap_ref_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_ref_1'], result.fe_params['pacmap_ref_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_ref_1'])), np.mean(list(df['pacmap_ref_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_ref_1']), list(df['pacmap_ref_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




model = smf.mixedlm("log_aire ~ pacmap_branche_1 + pacmap_branche_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_branche_1'] * np.array(df['pacmap_branche_1']) + result.params['pacmap_branche_2'] * np.array(df['pacmap_branche_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_branche_1'], result.fe_params['pacmap_branche_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_branche_1'])), np.mean(list(df['pacmap_branche_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_branche_1']), list(df['pacmap_branche_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




model = smf.mixedlm("log_aire ~ pacmap_loss_1 + pacmap_loss_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_loss_1'] * np.array(df['pacmap_loss_1']) + result.params['pacmap_loss_2'] * np.array(df['pacmap_loss_2']), np.array(df['log_aire']))[0]))

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_loss_1'], result.fe_params['pacmap_loss_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_loss_1'])), np.mean(list(df['pacmap_loss_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_loss_1']), list(df['pacmap_loss_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




################## MRI

df = pd.read_csv("/path/to/projection/espace_latent_IRM.csv")


model = smf.mixedlm("log_aire ~ pacmap_ref_1 + pacmap_ref_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_ref_1'] * np.array(df['pacmap_ref_1']) + result.params['pacmap_ref_2'] * np.array(df['pacmap_ref_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_ref_1'], result.fe_params['pacmap_ref_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_ref_1'])), np.mean(list(df['pacmap_ref_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_ref_1']), list(df['pacmap_ref_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




model = smf.mixedlm("log_aire ~ pacmap_branche_1 + pacmap_branche_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_branche_1'] * np.array(df['pacmap_branche_1']) + result.params['pacmap_branche_2'] * np.array(df['pacmap_branche_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_branche_1'], result.fe_params['pacmap_branche_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_branche_1'])), np.mean(list(df['pacmap_branche_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_branche_1']), list(df['pacmap_branche_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




model = smf.mixedlm("log_aire ~ pacmap_loss_1 + pacmap_loss_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pacmap_loss_1'] * np.array(df['pacmap_loss_1']) + result.params['pacmap_loss_2'] * np.array(df['pacmap_loss_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pacmap_loss_1'], result.fe_params['pacmap_loss_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pacmap_loss_1'])), np.mean(list(df['pacmap_loss_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pacmap_loss_1']), list(df['pacmap_loss_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()









# =============================================================================
# POSITION OF INCOHERENT SLICES IN THE LATENT SPACE
# =============================================================================


############# CT

df = pd.read_csv("/path/to/projections/espace_latent_CT.csv")

model = smf.mixedlm("log_aire ~ pca_branche_1 + pca_branche_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_branche_1'] * np.array(df['pca_branche_1']) + result.params['pca_branche_2'] * np.array(df['pca_branche_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_branche_1'], result.fe_params['pca_branche_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_branche_1'])), np.mean(list(df['pca_branche_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

# list of incoherent slices
incoherent_1 = []
incoherent_2 = []
incoherent_CT = ['p0008-slice072.jpg', 'p0014-slice053.jpg', 'p0014-slice062.jpg', 'p0014-slice066.jpg', 'p0017-slice033.jpg', 'p0017-slice034.jpg', 'p0021-slice050.jpg', 'p0023-slice065.jpg', 'p0027-slice034.jpg', 'p0027-slice068.jpg', 'p0027-slice069.jpg', 'p0027-slice070.jpg', 'p0027-slice071.jpg', 'p0027-slice072.jpg', 'p0040-slice016.jpg', 'p0040-slice069.jpg', 'p0040-slice073.jpg', 'p0040-slice075.jpg', 'p0048-slice019.jpg', 'p0048-slice020.jpg', 'p0048-slice021.jpg', 'p0048-slice022.jpg', 'p0048-slice023.jpg', 'p0048-slice068.jpg', 'p0048-slice070.jpg', 'p0065-slice072.jpg', 'p0065-slice080.jpg', 'p0065-slice082.jpg', 'p0066-slice066.jpg', 'p0066-slice069.jpg', 'p0066-slice071.jpg', 'p0066-slice073.jpg', 'p0101-slice062.jpg', 'p0101-slice063.jpg', 'p0101-slice064.jpg', 'p0101-slice065.jpg', 'p0104-slice007.jpg', 'p0106-slice070.jpg', 'p0106-slice086.jpg', 'p0118-slice066.jpg', 'p0125-slice059.jpg', 'p0132-slice044.jpg', 'p0132-slice047.jpg', 'p0150-slice071.jpg', 'p0157-slice030.jpg', 'p0157-slice032.jpg', 'p0162-slice080.jpg', 'p0165-slice000.jpg', 'p0165-slice001.jpg', 'p0165-slice002.jpg', 'p0166-slice063.jpg', 'p0166-slice064.jpg', 'p0166-slice065.jpg', 'p0166-slice066.jpg', 'p0166-slice067.jpg', 'p0166-slice068.jpg', 'p0166-slice081.jpg', 'p0166-slice082.jpg', 'p0166-slice083.jpg', 'p0166-slice086.jpg', 'p0166-slice087.jpg', 'p0183-slice009.jpg', 'p0183-slice011.jpg', 'p0183-slice012.jpg', 'p0183-slice023.jpg', 'p0183-slice062.jpg', 'p0183-slice063.jpg', 'p0183-slice066.jpg', 'p0183-slice067.jpg', 'p0183-slice068.jpg', 'p0185-slice032.jpg', 'p0186-slice070.jpg', 'p0186-slice078.jpg', 'p0186-slice082.jpg', 'p0186-slice085.jpg', 'p0192-slice062.jpg', 'p0192-slice063.jpg', 'p0192-slice064.jpg', 'p0192-slice066.jpg', 'p0198-slice008.jpg', 'p0198-slice009.jpg', 'p0198-slice010.jpg', 'p0198-slice011.jpg', 'p0198-slice012.jpg', 'p0208-slice027.jpg', 'p0208-slice067.jpg', 'p0208-slice079.jpg', 'p0213-slice029.jpg', 'p0213-slice031.jpg', 'p0216-slice035.jpg', 'p0216-slice036.jpg', 'p0216-slice037.jpg', 'p0216-slice038.jpg', 'p0216-slice051.jpg', 'p0216-slice057.jpg', 'p0216-slice058.jpg', 'p0216-slice059.jpg', 'p0216-slice060.jpg', 'p0216-slice070.jpg', 'p0219-slice081.jpg', 'p0221-slice025.jpg', 'p0221-slice027.jpg', 'p0221-slice052.jpg', 'p0221-slice053.jpg', 'p0221-slice054.jpg', 'p0221-slice055.jpg', 'p0226-slice054.jpg', 'p0238-slice051.jpg', 'p0238-slice083.jpg', 'p0238-slice084.jpg', 'p0238-slice085.jpg', 'p0238-slice086.jpg', 'p0238-slice087.jpg', 'p0257-slice056.jpg', 'p0257-slice060.jpg', 'p0262-slice035.jpg', 'p0262-slice037.jpg', 'p0262-slice038.jpg', 'p0266-slice036.jpg', 'p0266-slice066.jpg', 'p0266-slice069.jpg', 'p0266-slice071.jpg', 'p0269-slice067.jpg', 'p0274-slice028.jpg', 'p0274-slice029.jpg', 'p0274-slice030.jpg', 'p0274-slice031.jpg', 'p0274-slice050.jpg', 'p0274-slice052.jpg', 'p0274-slice053.jpg', 'p0284-slice068.jpg', 'p0285-slice040.jpg', 'p0286-slice079.jpg', 'p0286-slice080.jpg', 'p0286-slice081.jpg', 'p0286-slice082.jpg', 'p0286-slice083.jpg', 'p0286-slice088.jpg', 'p0287-slice002.jpg', 'p0287-slice003.jpg', 'p0287-slice004.jpg', 'p0288-slice000.jpg', 'p0288-slice078.jpg', 'p0292-slice024.jpg', 'p0292-slice025.jpg', 'p0292-slice026.jpg', 'p0292-slice027.jpg', 'p0292-slice043.jpg', 'p0292-slice044.jpg', 'p0292-slice046.jpg', 'p0297-slice033.jpg', 'p0298-slice032.jpg', 'p0298-slice036.jpg', 'p0298-slice037.jpg', 'p0298-slice042.jpg', 'p0305-slice030.jpg', 'p0305-slice031.jpg', 'p0305-slice058.jpg', 'p0305-slice059.jpg', 'p0306-slice030.jpg', 'p0308-slice056.jpg', 'p0308-slice057.jpg', 'p0036-slice028.jpg', 'p0113-slice062.jpg', 'p0175-slice076.jpg', 'p0189-slice010.jpg', 'p0189-slice025.jpg', 'p0189-slice027.jpg', 'p0189-slice028.jpg', 'p0200-slice061.jpg', 'p0224-slice036.jpg', 'p0224-slice037.jpg', 'p0224-slice039.jpg', 'p0013-slice059.jpg', 'p0026-slice043.jpg', 'p0026-slice044.jpg', 'p0026-slice046.jpg', 'p0026-slice048.jpg', 'p0026-slice049.jpg', 'p0026-slice050.jpg', 'p0026-slice051.jpg', 'p0026-slice052.jpg', 'p0026-slice057.jpg', 'p0071-slice072.jpg', 'p0071-slice074.jpg', 'p0071-slice076.jpg', 'p0182-slice089.jpg', 'p0217-slice044.jpg', 'p0217-slice068.jpg', 'p0263-slice067.jpg', 'p0263-slice068.jpg', 'p0291-slice029.jpg']
for c in incoherent_CT:
    incoherent_1.append(df.loc[df['coupes'] == c, 'pca_branche_1'].iloc[0])
    incoherent_2.append(df.loc[df['coupes'] == c, 'pca_branche_2'].iloc[0])


plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_branche_1']), list(df['pca_branche_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.scatter(incoherent_1, incoherent_2, s=100, marker='*', c='skyblue')
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()



############## MRI

df = pd.read_csv("/path/to/projections/espace_latent_IRM.csv")

model = smf.mixedlm("log_aire ~ pca_branche_1 + pca_branche_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_branche_1'] * np.array(df['pca_branche_1']) + result.params['pca_branche_2'] * np.array(df['pca_branche_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_branche_1'], result.fe_params['pca_branche_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_branche_1'])), np.mean(list(df['pca_branche_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

# list of incoherent slices
incoherent_1 = []
incoherent_2 = []
incoherent_IRM = ['p0027-slice015.jpg', 'p0027-slice016.jpg', 'p0027-slice017.jpg', 'p0101-slice017.jpg', 'p0104-slice013.jpg', 'p0149-slice023.jpg', 'p0165-slice003.jpg', 'p0165-slice004.jpg', 'p0166-slice017.jpg', 'p0183-slice003.jpg', 'p0183-slice017.jpg', 'p0183-slice018.jpg', 'p0192-slice016.jpg', 'p0198-slice006.jpg', 'p0216-slice015.jpg', 'p0219-slice019.jpg', 'p0221-slice007.jpg', 'p0221-slice020.jpg', 'p0238-slice020.jpg', 'p0266-slice020.jpg', 'p0274-slice010.jpg', 'p0287-slice001.jpg', 'p0287-slice017.jpg', 'p0288-slice005.jpg', 'p0288-slice008.jpg', 'p0292-slice007.jpg', 'p0292-slice012.jpg', 'p0292-slice015.jpg', 'p0299-slice003.jpg', 'p0305-slice010.jpg', 'p0305-slice017.jpg', 'p0306-slice010.jpg', 'p0180-slice016.jpg', 'p0224-slice019.jpg', 'p0224-slice020.jpg', 'p0026-slice013.jpg', 'p0263-slice009.jpg', 'p0263-slice018.jpg']
for c in incoherent_IRM:
    incoherent_1.append(df.loc[df['coupes'] == c, 'pca_branche_1'].iloc[0])
    incoherent_2.append(df.loc[df['coupes'] == c, 'pca_branche_2'].iloc[0])


plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_branche_1']), list(df['pca_branche_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.scatter(incoherent_1, incoherent_2, s=100, marker='*', c='skyblue')
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




# =============================================================================
# COMPARISON OF MODELS TRAINE WITH AND WIHOUT UNCOHERENT SLICES
# =============================================================================

################## CT


df = pd.read_csv("/paht/to/projection/espace_latent_coherent_CT.csv")


model = smf.mixedlm("log_aire ~ pca_ref_1 + pca_ref_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_ref_1'] * np.array(df['pca_ref_1']) + result.params['pca_ref_2'] * np.array(df['pca_ref_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_ref_1'], result.fe_params['pca_ref_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_ref_1'])), np.mean(list(df['pca_ref_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_ref_1']), list(df['pca_ref_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()



model = smf.mixedlm("log_aire ~ pca_coherent_1 + pca_coherent_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_coherent_1'] * np.array(df['pca_coherent_1']) + result.params['pca_coherent_2'] * np.array(df['pca_coherent_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_coherent_1'], result.fe_params['pca_coherent_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_coherent_1'])), np.mean(list(df['pca_coherent_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_coherent_1']), list(df['pca_coherent_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()




################## IRM


df = pd.read_csv("/path/to/projection/espace_latent_coherent_IRM.csv")


model = smf.mixedlm("log_aire ~ pca_ref_1 + pca_ref_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_ref_1'] * np.array(df['pca_ref_1']) + result.params['pca_ref_2'] * np.array(df['pca_ref_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_ref_1'], result.fe_params['pca_ref_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_ref_1'])), np.mean(list(df['pca_ref_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_ref_1']), list(df['pca_ref_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()



model = smf.mixedlm("log_aire ~ pca_coherent_1 + pca_coherent_2", df, groups=df['patient'])
result = model.fit()
print(result.summary())
print("Spearman", abs(spearmanr(result.params['pca_coherent_1'] * np.array(df['pca_coherent_1']) + result.params['pca_coherent_2'] * np.array(df['pca_coherent_2']), np.array(df['log_aire']))[0]))
# print(result.cov_re.iloc[0, 0])

b0, b1, b2 =result.fe_params['Intercept'], result.fe_params['pca_coherent_1'], result.fe_params['pca_coherent_2']
a = b2/b1
x_center, y_center = np.mean(list(df['pca_coherent_1'])), np.mean(list(df['pca_coherent_2']))
t = np.linspace(-1, 1, 100)
x_line = x_center + t 
y_line = y_center + t * a

plt.figure(figsize=(16,10))
plt.scatter(list(df['pca_coherent_1']), list(df['pca_coherent_2']), c=df['log_aire'], s=30, alpha=0.8, cmap=custom_cmap)
plt.plot(x_line, y_line, color="navy", linewidth=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
cbar.set_label('log(lesion area (mm²))', size=30)
plt.xlabel('dim1', fontsize = 30)
plt.ylabel('dim2', fontsize = 30)
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.show()
