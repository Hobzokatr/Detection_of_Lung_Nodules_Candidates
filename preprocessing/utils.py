import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from glob import glob
import csv
import cv2
from random import randint
import random
import numpy as np
from load_tools import load_itk, show_images, world_2_voxel
import SimpleITK as sitk
import matplotlib as plt


def visualize_data():
    all_candidates = pd.read_csv('/datagrid/Medical/archive/archive_dirs/nodules/Luna/candidates.csv')
    nonnodules = all_candidates[all_candidates['class']==0]  
    nodules = all_candidates[all_candidates['class'] == 1]
    joined_nodules = nonnodules.append(nodules)

    '(nodules: {} non-nodules: {}) / {}'.format(nodules.shape[0],nonnodules.shape[0],all_candidates.shape[0])
    num_images_show = 18

    width_size = 32
    l = []
    patches = []
    for i in range(num_images_show):
        im = nodules.iloc[i]
        for subset in range(10):
            input_path = "/datagrid/Medical/archive/archive_dirs/nodules/Luna/subset{}".format(subset)
            im_path = join(input_path, im['seriesuid']+'.mhd')
            if isfile(im_path): 
                lung_img = sitk.GetArrayFromImage(sitk.ReadImage(im_path))
                _, orig, spac = load_itk(im_path)
                vox_coords = world_2_voxel([float(im['coordZ']), float(im['coordY']), float(im['coordX'])], orig, spac)
                y_class = int(im['class'])
                w = width_size / 2
                patch = lung_img[int(vox_coords[0]-1): int(vox_coords[0]+2),
                        int(vox_coords[1] - w): int(vox_coords[1] + w),
                        int(vox_coords[2] - w): int(vox_coords[2] + w)]
                l.append(patch[0])
                patches.append(patch)
        print('l[i]',l)
        plt.pyplot.imsave("image_{}.png".format(i), l[i], cmap='gray')
    show_images(l, 3)
    print("Done")
