#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:04:37 2022

@author: moreau
"""



import glob

# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F



# numpy and pandas
import numpy
import pandas as pd

# Common python packages
import os

#monai stuff
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose
from monai.data import CacheDataset

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure



# #################################################################################################################################load data
def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=/frac{2|A/cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))

    intersection = numpy.count_nonzero(result & reference)

    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the inumpyut rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def ravd(result, reference):
    """
    Relative absolute volume difference.

    Compute the relative absolute volume difference between the (joined) binary objects
    in the two images.

    Parameters
    ----------
    result : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Inumpyut data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    ravd : float
        The relative absolute volume difference between the object(s) in ``result``
        and the object(s) in ``reference``. This is a percentage value in the range
        :math:`[-1.0, +inf]` for which a :math:`0` denotes an ideal score.

    Raises
    ------
    RuntimeError
        If the reference object is empty.

    See also
    --------
    :func:`dc`
    :func:`precision`
    :func:`recall`

    Notes
    -----
    This is not a real metric, as it is directed. Negative values denote a smaller
    and positive values a larger volume than the reference.
    This implementation does not check, whether the two supplied arrays are of the same
    size.

    Examples
    --------
    Considering the following inumpyuts

    >>> import numpy
    >>> arr1 = numpy.asarray([[0,1,0],[1,1,1],[0,1,0]])
    >>> arr1
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> arr2 = numpy.asarray([[0,1,0],[1,0,1],[0,1,0]])
    >>> arr2
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])

    comparing `arr1` to `arr2` we get

    >>> ravd(arr1, arr2)
    -0.2

    and reversing the inumpyuts the directivness of the metric becomes evident

    >>> ravd(arr2, arr1)
    0.25

    It is important to keep in mind that a perfect score of `0` does not mean that the
    binary objects fit exactely, as only the volumes are compared:

    >>> arr1 = numpy.asarray([1,0,0])
    >>> arr2 = numpy.asarray([0,0,1])
    >>> ravd(arr1, arr2)
    0.0

    """
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))

    vol1 = numpy.count_nonzero(result)
    vol2 = numpy.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2)

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the inumpyut has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def evaluate_generator(generator, train_loader, test_files):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating Mask-w images

    """
    
    res_train, res_test = [], []
    res_tot = {}

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    with torch.no_grad():


        dice=[]
        dico_dice= {}
        HD = []
        dico_hd = {}
        RAVD = []
        dico_raad = {}

        dice_pred=[]
        RAVD_pred=[]
        for i, batch in enumerate(test_loader):
            
            coupe = test_files[i]['cerveau'].split('/')[-1][:-4]
            
            # Inumpyuts CT-w and Mask-w
            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = batch["GT"].type(Tensor) 
            
            fake_Mask = generator(real_CT) # for baseline U_Net and output constrained U-Net
            # fake_Mask, = batch["GT"].type(Tensor) # for feature constrained U-Net
            
            real_Mask = Tensor.cpu(real_Mask).numpy()
            
            fake_Mask = torch.argmax(fake_Mask, dim=1).reshape((fake_Mask.shape[0],1,192,192))
            fake_Mask = Tensor.cpu(fake_Mask).numpy()
            real_CT = Tensor.cpu(real_CT).numpy()


            if numpy.count_nonzero(fake_Mask) != 0 and numpy.count_nonzero(real_Mask) != 0: 

                
                dice = dc(fake_Mask, real_Mask)
                dico_dice[coupe] = dc(fake_Mask, real_Mask)
                HD = hd(fake_Mask,real_Mask)
                dico_hd[coupe] = hd(fake_Mask,real_Mask)
                RAVD = ravd(fake_Mask,real_Mask)
                dico_raad[coupe] = ravd(fake_Mask,real_Mask)

            else :
                dice= 0
                dico_dice[coupe] = 0
                HD = 130
                dico_hd[coupe] = 130
                RAVD = -1
                dico_raad[coupe] = -1
                
                   
            res_tot[test_files[i]['cerveau'].split('/')[-1]] = [dice, HD, RAVD]
            res_test.append([dice, HD, RAVD])

        
        df = pd.DataFrame([
            pd.DataFrame(res_test, columns=['DICE', 'HD', 'RAVD']).mean().squeeze(),
            pd.DataFrame(res_test, columns=['DICE', 'HD', 'RAVD']).std().squeeze()
        ], index=['Training set', 'Test set']).T
        
    return nb_detection/nb_images, df, res_tot, dico_dice, dico_hd, dico_raad
        




KEYS = ("cerveau", "GT")


xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 1


############## test

def split_into_five_parts(data):
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
    subsets = split_into_five_parts(data)
    subset_k = subsets[k - 1]
    others = [item for i, subset in enumerate(subsets) if i != (k - 1) for item in subset]
    return subset_k


from generator_UNet import UNet # for baseline U-Net and output constrained U-Net
# from regressor_UNet import UNet # for feature constrained U-Net


df = {'detection': [], 'Dice': [], 'HD':[], 'RAAD':[]}

detec = []
dsc = []
h = []
raad = []

for fold in range(1,6):

    test_files = []
    test_dir = "/path/to/data/image/total/"
    test_dir_label = "/path/to/data/ref/total/"
    

    
    images_ = sorted(glob.glob(test_dir + "*.jpg")) 
    labels_ = sorted(glob.glob(test_dir_label + "*.png"))
    
    images = []
    labels = []
    for i in range(len(images_)):
        images.append(images_[i])
        labels.append(labels_[i])

    patients = sorted(os.listdir("/path/to/3D/data/image/total")) # to get the list of all patients

    # separation in folder
    test_pat = get_subset_and_others(patients, fold)
    
    test_images, test_labels = [], []
    for i in images:
        if i[-18:-13]+'.nii.gz' in test_pat:
            test_images.append(i)
    for i in labels:
        if i[-18:-13]+'.nii.gz' in test_pat:
            test_labels.append(i)

    test_files = []
    for im in range(len(test_images)):
        test_files.append({"cerveau": test_images[im], "GT": test_labels[im]})

    # load dataset
    test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    
    for i, batch in enumerate(test_ds):
        batch['cerveau']=batch['cerveau'][0:1,:,:]
        batch['GT']=batch['GT'][0:1,:,:]
        I = batch['GT']>1
        batch['GT'] = I.long()
        


    generator = UNet()
    generator.cuda()
    generator.load_state_dict(torch.load("/path/to/model/weights/fold"+str(fold)+"/checkpoints/checkpoint.pth"))
    generator.eval()
    detection, res, listes, dico_dice_ref, dico_hd_ref, dico_raad_ref, dico_mse = evaluate_generator(generator, test_loader, test_files)

    dsc = dsc + list(dico_dice_ref.values())
    h = h + list(dico_hd_ref.values())
    raad = raad + list(dico_raad_ref.values())
    

print(numpy.mean(dsc), numpy.std(dsc))
print(numpy.mean(h), numpy.std(h))
print(numpy.mean(raad), numpy.std(raad))

