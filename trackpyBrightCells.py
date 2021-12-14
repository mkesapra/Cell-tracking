#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:01:34 2020

@author: kesaprm
"""


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

frames = pims.open('images/20X MQAE 7.5mM well 1 .PNG')

micron_per_pixel =0.15192872980868# 0.09#0.15192872980868
feature_diameter = 2.12#2.48 # um
radius = int(np.round(feature_diameter/2.0/micron_per_pixel))
if radius % 2 == 0:
    radius += 1
print('Using a radius of {:d} px'.format(radius))
frames[0]
#radius = 13

f_locate = tp.locate(frames[0], radius+2, minmass=500)
tp.annotate(f_locate, frames[0], plot_style={'markersize': radius});

plt.figure()
tp.annotate(f_locate, frames[0], plot_style={'markersize': radius*2}, ax=plt.gca())
plt.ylim(0, 1000)
plt.xlim(0, 1000);

# f_bf = tp.locate_brightfield_ring(frames[0], 2.0*radius+1)
# tp.annotate(f_bf, frames[0], plot_style={'markersize': radius});

# plt.figure()
# tp.annotate(f_bf, frames[0], plot_style={'markersize': radius*2}, ax=plt.gca())
# plt.ylim(0, 1000)
# plt.xlim(0, 1000);

# f_bf_with_prev = tp.locate_brightfield_ring(frames[0], 2.0*radius+1, previous_coords=f_locate)


plot_properties = dict(markeredgewidth=2, markersize=radius*2, markerfacecolor='none')

plt.figure()
plt.imshow(frames[0])
plt.plot(f_locate['x'], f_locate['y'], 'o', markeredgecolor='red', **plot_properties)
#plt.plot(f_bf['x'], f_bf['y'], 'o', markeredgecolor='green', **plot_properties)
#plt.plot(f_bf_with_prev['x'], f_bf_with_prev['y'], 'x', markeredgecolor='blue', **plot_properties)

plt.ylim(0, 1000)
plt.xlim(0, 1000);

#linking particle positions

trajectories = []

# Take only the x and y positions from the standard `locate` function
previous_coords = f_locate[['x', 'y']].copy()

# Store the radius
previous_coords['r'] = radius

# Number the particles, this will get copied for all frames
previous_coords['particle'] = np.arange(0, len(f_locate), 1)

for frame in frames:
    # The radius of the particle is fixed, but the radius of the feature can change based on focal plane
    average_radius = np.mean(previous_coords['r'])
    
    # Locate particles in this frame based on the previous positions
    f_bf = tp.locate_brightfield_ring(frame, 2.0*average_radius+1, previous_coords=previous_coords)
    
    # Store the result
    trajectories.append(f_bf)
    
    # Store the previous positions
    previous_coords = f_bf

# Merge all into one dataframe
trajectories = pd.concat(trajectories, sort=True)

# Inspect the position in the first 10 frames for particle 0
print(trajectories[trajectories['particle']==0].head(10))

# Overlay trajectories on average of all frames
background = np.mean(frames, axis=0)
tp.plot_traj(trajectories, superimpose=background, label=True, plot_style={'linewidth': 2});