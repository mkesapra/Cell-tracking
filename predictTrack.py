#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:45:34 2020

@author: kesaprm
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trackpy as tp
import pims

class SimulatedFrame(object):
    
    def __init__(self, shape, dtype=np.uint8):
        self.image = np.zeros(shape, dtype=dtype)
        self._saturation = np.iinfo(dtype).max
        self.shape = shape
        self.dtype =dtype
        
    def add_spot(self, pos, amplitude, r, ecc=0):
        "Add a Gaussian spot to the frame."
        x, y = np.meshgrid(*np.array(list(map(np.arange, self.shape))) - np.asarray(pos))
        spot = amplitude*np.exp(-((x/(1 - ecc))**2 + (y*(1 - ecc))**2)/(2*r**2)).T
        self.image += np.clip(spot, 0, self._saturation).astype(self.dtype)
        
    def with_noise(self, noise_level, seed=0):
        "Return a copy with noise."
        rs = np.random.RandomState(seed)
        noise = rs.randint(-noise_level, noise_level, self.shape)
        noisy_image = np.clip(self.image + noise, 0, self._saturation).astype(self.dtype)
        return noisy_image
    
    def add_noise(self, noise_level, seed=0):
        "Modify in place with noise."
        self.image = self.with_noise(noise_level, seed=seed)
        
fig, axes = plt.subplots(2, 2)

frame = SimulatedFrame((20, 30))
frame.add_spot((10, 15), 200, 2.5)

for ax, noise_level in zip(axes.ravel(), [1, 20, 40, 90]):
    noisy_copy = frame.with_noise(noise_level)
    features = tp.locate(noisy_copy, 13, topn=1, engine='python')
    tp.annotate(features, noisy_copy, plot_style=dict(marker='x'), imshow_style=dict(vmin=0, vmax=255), ax=ax)
    dx, dy, ep = features[['x', 'y', 'ep']].iloc[0].values - [16, 10, 0]
    ax.set(xticks=[5, 15, 25], yticks=[5, 15])
    ax.set(title=r'Signal/Noise = {signal}/{noise}'.format(
              signal=200, noise=noise_level))
    ax.text(0.5, 0.1, r'$\delta x={dx:.2}$  $\delta y={dy:.2}$'.format(
                dx=abs(dx), dy=abs(dy)), ha='center', color='white', transform=ax.transAxes)
    ax.text(0.05, 0.85, r'$\epsilon={ep:.2}$'.format(ep=abs(ep)), ha='left', color='white',
            transform=ax.transAxes)
fig.subplots_adjust()