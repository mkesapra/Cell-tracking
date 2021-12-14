#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:39:33 2020

@author: kesaprm
"""
import numpy as np
import pandas as pd

import pims
import trackpy as tp
import os

import matplotlib  as mpl 
import matplotlib.pyplot as plt 

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

datapath = 'images/'
id_example = 0
rawframes = pims.ImageSequence(os.path.join(datapath + '*.tif'))
plt.imshow(rawframes[id_example]);

def crop(img):
    """
    Crop the image to select the region of interest
    """
    x_min = 45
    x_max = -35
    y_min = 100
    y_max = -300 
    return img[y_min:y_max,x_min:x_max]

rawframes = pims.ImageSequence(os.path.join(datapath + '*.tif'))
plt.imshow(rawframes[id_example]);