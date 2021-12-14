#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:01:34 2020

@author: kesaprm
"""


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp


### Step-1: Read the data
frames = pims.open('images/*.tif')

### Step-2: Locate Features
f=tp.locate(frames[0],99)
f.head()
tp.annotate(f,frames[0])

fig,ax = plt.subplots()
ax.hist(f['mass'],bins=10)
ax.set(xlabel = 'mass', ylabel = 'count')

f = tp.locate(frames[0],99,minmass=6250)
tp.annotate(f, frames[0])

tp.subpx_bias(f)

f= tp.batch(frames[:],99,minmass=6250)

### Step- 3: Link features to trajectories
t= tp.link(f,2,memory=3)

t.head()

t1= tp.filter_stubs(t,1)

print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

plt.figure()
tp.mass_size(t1.groupby('particle').mean())

t2 = t1[((t1['mass'] > 1) )]
plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0]);

plt.figure()
tp.plot_traj(t2);

d = tp.compute_drift(t2)
d.plot()
plt.show()

tm = tp.subtract_drift(t2.copy(), d)
ax = tp.plot_traj(tm)
plt.show()


im = tp.imsd(tm, 100/285., 24)  # microns per pixel = 100/285., frames per second = 24
fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')


em = tp.emsd(tm, 100/285., 24) # microns per pixel = 100/285., frames per second = 24
fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set(ylim=(1e-2, 10));

plt.figure()
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$');
tp.utils.fit_powerlaw(em) 

with tp.PandasHDFStore('data.h5') as s:
    tp.batch(frames, 11, invert=True, minmass=200, output=s)
    
with tp.PandasHDFStore('data.h5') as s:
    # As before, we require a minimum "life" of 5 frames and a memory of 3 frames
    for linked in tp.link_df_iter(s, 2, memory=3):
        s.put(linked)

with tp.PandasHDFStore('data.h5') as s:
    trajectories = pd.concat(iter(s))
    
plt.figure()
background = np.mean(frames, axis=0)
tp.plot_traj(trajectories, superimpose=background, plot_style={'linewidth': 7})
plt.ylim(0, 1000)
plt.xlim(0, 1000);

