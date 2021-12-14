#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:29:35 2021

@author: Manasa Kesapragada
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series 

import pims
import trackpy as tp

### Step-1: Read the data
frames = pims.open('/Users/kesaprm/FY19_20/Spring2020/Project/FS-Tracer_Data/M1_M2_Galvanotaxis_new/M2EF2/*.tif')


### Uncomment the below line to plot the image
plt.imshow(frames[0]);

### Step-2: Locate Features
f=tp.locate(frames[0],51)
#f.head() #to check the first 5 rows of f
#tp.annotate(f,frames[0])

### Step-3: Refine parameters
### Uncomment the below lines to check the mass histogram
# fig,ax = plt.subplots()
# ax.hist(f['mass'],bins=20)
# # Optionally, label the axes.
# ax.set(xlabel = 'mass', ylabel = 'count')

# Here I am trying to change the minmass looking at the mass-count plot above. 
# Decreasing the minmass value includes more cells, we could find this from the f dataframe
# 51 acts like a threshold 

# Lets assign the finalized threshold and minmass in variables so that they could be used in 
# multiple places
thresh = 51
min_mass =1.4*1e5
f = tp.locate(frames[0],thresh)#,minmass = min_mass)
### Uncomment the below line to plot the segmented image
tp.annotate(f, frames[0], plot_style={'markersize': 8})


#checking for subpixel accuracy
#tp.subpx_bias(f)


### Step-4: Locate features in all frames
f = tp.batch(frames[:], thresh)#,minmass=min_mass);


###Step-5,6: Link features into particle trajectories
# 60 is the no. of pixels moved and memory keeps track of disappeared particles and
# maintains their ID for up to some number of frames after their last appearance
t= tp.link(f,55,memory=1)

###Step-7: Filter trajectories that exist in all the frames
t1 = tp.filter_stubs(t, len(frames))
print('No. of tracks before filtering:', t['particle'].nunique())
print('No. of tracks after filtering:', t1['particle'].nunique())




###Step-8: Results - Trajectories plotting
background = np.mean(frames, axis=0)

### Uncomment the below lines to plot the trajectories
plt.figure()
#superimpose will help to plot the trajectories on the background image
tp.plot_traj(t1,superimpose=background,ax=plt.gca(),  plot_style={'linewidth': 2}); #label=True,
plt.show()

#### Below code is for reference. Commented as not used currently
#plt.rcParams['font.size'] = '8'
# plt.imshow(frames[0])
# plt.quiver(t1.x, t1.y, t1.x, -t1.y, pivot='middle', headwidth=4, headlength=6, color='red')

#plt.xlim(1000, 2000)
#plt.ylim(1000, 2000);

###Step-9: Results - Convert x,y to relative values

#Convert the x, y values from pixels to micrometers -- 1 Pixels= 264.5833 Micrometers
#Correction- Refer Yao-Hui email Dt:Sep-14, the resolution in all images is 4.31 pix per micron, so 1 pixel = 1/4.31 microns
##generate relative x,y-values
all_rel_x = []
all_rel_y = []
for k in t1.particle.unique():
    curr_cell_x = t1.x[t1.particle == k]*(1/4.31) #2.155 for 10x, 4.31 for 20x
    curr_cell_y = t1.y[t1.particle == k]*(1/4.31) #2.155 for 10x ,4.31 for 20x
    rel_x = []; rel_y =[];
    for i in range(0 ,len(frames)-1): 
        rel_x.append(curr_cell_x[i] - curr_cell_x[0])
        rel_y.append(curr_cell_y[i] - curr_cell_y[0])
    all_rel_x.append(rel_x) 
    all_rel_y.append(rel_y)




###Step-10: Results - Plot the relative x, y values
# Uncomment the below lines when needed to plot
plt.figure()
for j in range(0,  len(all_rel_x)):
    if(all(i <= 0 for i in all_rel_x[j])):
        plt.plot(all_rel_x[j],all_rel_y[j],'r-', linewidth=0.5)
    else:
        plt.plot(all_rel_x[j],all_rel_y[j],'r-' ,linewidth=0.5)
plt.xlabel(r'x [microns]')
plt.ylabel('y [microns]')
plt.title('Relative trajectories')
plt.show()

##Directedness calculation
directedness = []
for k in range(0, len(all_rel_x)):
    d_in_loop = []
    for i in range(1, len(all_rel_x[k])): #first val is 0, hence the range starts from 1
        x_val = all_rel_x[k][i]
        y_val = all_rel_y[k][i]
        euc_d = np.sqrt(x_val**2 + y_val**2)
        d_in_loop.append(x_val/ euc_d)
    directedness.append(d_in_loop)
    
###Step-11: Results - Plot the Directedness
avgD = []
for m in range(0, len(directedness)):
    avgD.append(np.sum(directedness[m])/len(directedness[m]))

# Bar plot for directedness
for k in range(0, len(all_rel_x)): 
    if (avgD[k] < 0):
        colors = 'k'
    else:
        colors = 'r'
    plt.bar(k,avgD[k], color = colors)   
plt.xlabel("Cells");
plt.ylabel("Directedness")
plt.title("Avg Directedness of the cells")
plt.show()


###Distance calculation

distance = []
for k in range(0, len(all_rel_x)):
    dist_in_loop = []
    for i in range(0, len(all_rel_x[k])-1): #first val is 0, hence the range starts from 1
        x0 = all_rel_x[k][i]
        x1 = all_rel_x[k][i+1]
        y0 = all_rel_y[k][i]
        y1 = all_rel_y[k][i+1]
        dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        dist_in_loop.append(dist)
    distance.append(dist_in_loop)
 
plt.figure()
for j in range(0,  len(all_rel_x)):
        plt.plot(range(0,  len(all_rel_x[k])-1),distance[j])
plt.xlabel(r'time in frames')
plt.ylabel('Distance travelled by a cell[microns]')
plt.title('Distance')
plt.show()

##Displacement calculation

displacement =[]
for k in range(0, len(all_rel_x)):
    x0 = all_rel_x[k][0]
    xN = all_rel_x[k][len(all_rel_x[k])-1]
    y0 = all_rel_y[k][0]
    yN = all_rel_y[k][len(all_rel_x[k])-1]
    disp = np.sqrt((xN-x0)**2 + (yN-y0)**2)
    displacement.append(disp)

plt.figure()

plt.bar(range(0,  len(all_rel_x)),displacement)
plt.xlabel(r'No. of cells')
plt.ylabel('Cell Displacement[microns]')
plt.title('Cell Displacement')
plt.show()


plt.boxplot(displacement)
plt.xticks([1], ['Avg. displacement of the cells'])
# ##Save all the values to export in a dataframe
# d = {'CellNo': range(1,len(all_rel_x)+1),'X': all_rel_x , 'Y': all_rel_y , 'Directedness': directedness}
# df_final =  pd.DataFrame(data=d)

###Step-12: Export the x, y relative values and directedness to a csv
# df_final.to_csv(r'Image_Analysis_vals.csv',index=None)
# print('X,Y values in micrometers and Directedness values are written to the CSV File successfully.')


