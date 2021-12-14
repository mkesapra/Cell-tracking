
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:29:35 2021

@author: kesaprm
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

### Step-1: Read the data
#frames = pims.open('/Users/kesaprm/Downloads/Separated_Data/No EF3/Bottom Middle/phc/*.tif')

frames = pims.open('/Users/kesaprm/FY19_20/Spring2020/Project/FS-Tracer_Data/Codys/Macrophage_plate1_9_12_2021_Plate_R_p00_0_H12f25d4.png')
### Show the first frame
plt.imshow(frames[0]);

### Step-2: Locate Features
f=tp.locate(frames[0],71)
#f.head() #to check the first 5 rows of f
tp.annotate(f,frames[0])

#Refine parameters
fig,ax = plt.subplots()
ax.hist(f['mass'],bins=20)
# Optionally, label the axes.
ax.set(xlabel = 'mass', ylabel = 'count')

# Here I am trying to change the minmass looking at the mass-count plot above. 
# Decreasing the minmass value includes more cells
# 37 acts like a threshold 
f = tp.locate(frames[0],51,minmass=6e4,invert=True)
tp.annotate(f, frames[0], plot_style={'markersize': 10})

#checking for subpixel accuracy

tp.subpx_bias(f)


# Locate features in all frames
f = tp.batch(frames[:],51,minmass=6e4)

#Step 3: Link features into particle trajectories
# 60 is the no. of pixels moved and memory keeps track of disappeared particles and
# maintains their ID for up to some number of frames after their last appearance
t= tp.link(f,60,memory=1)

#Filter trajectories that exist in all the frames
t1 = tp.filter_stubs(t, len(frames))
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())


# plt.figure()
# tp.mass_size(t1.groupby('particle').mean());

# t2 = t1[(t1['ecc'] > 0.01)]

# plt.figure()
# tp.annotate(t2[t2['frame'] == 0], frames[0]);

#Trajectories plotting
plt.figure()
tp.plot_traj(t);

#Remove drift
d = tp.compute_drift(t1)

d.plot()
plt.show()

tm = tp.subtract_drift(t1.copy(), d)
ax = tp.plot_traj(tm)
plt.show()



#Trajectories plotting
background = np.mean(frames, axis=0)
plt.figure()
tp.plot_traj(t1,superimpose=background,ax=plt.gca(),  plot_style={'linewidth': 2}); #label=True,
#plt.rcParams['font.size'] = '8'
# plt.imshow(frames[0])
# plt.quiver(t1.x, t1.y, t1.x, -t1.y, pivot='middle', headwidth=4, headlength=6, color='red')

plt.xlim(0, 1000)
plt.ylim(1000, 2000);

all_rel_x = []
all_rel_y = []
for k in t1.particle.unique():
    curr_cell_x = t1.x[t1.particle == k]
    curr_cell_y = t1.y[t1.particle == k]
    rel_x = []; rel_y =[];
    for i in range(0 ,len(frames)-1): 
        rel_x.append(curr_cell_x[i] - curr_cell_x[0])
        rel_y.append(curr_cell_y[i] - curr_cell_y[0])
    all_rel_x.append(rel_x) 
    all_rel_y.append(rel_y)
    #t1['rel_x'][t1.particle == k] = rel_x
    #t1['rel_y'][t1.particle == k] = rel_y 
       
        # t1.rel_y[t1.particle == k][i] = t1.rel_y[t1.particle == k][i] - t1.rel_y[t1.particle == k][0]
        # t1.rel_x[t1.particle == k].iat[i] = t1.rel_x[t1.particle == k].iat[i] - t1.rel_x[t1.particle == k].iat[0]
        # t1.rel_x[t1.particle == k].iat[i] = t1.rel_x[t1.particle == k][i] - t1.rel_x[t1.particle == k][0]
        # t1.rel_y[t1.particle == k][i] = t1.rel_y[t1.particle == k][i] - t1.rel_y[t1.particle == k][0]
        #t1['rel_y'][t1.particle == k][i] = rel_y 

###Step-10: Results - Plot the relative x, y values
for j in range(0,  len(all_rel_x)):
    if(all(i <= 0 for i in all_rel_x[j])):
        plt.plot(all_rel_x[j],all_rel_y[j],'r-', linewidth=0.5)
    else:
        plt.plot(all_rel_x[j],all_rel_y[j],'r-' ,linewidth=0.5)
plt.xlabel('x[px]')
plt.ylabel('y[px]')

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
    
#Avg Directedness for plotting
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



# writing to Excel
datatoexcel = pd.ExcelWriter('trajectories.xlsx')
  
# write DataFrame to excel
t1.to_excel(datatoexcel)
  
# save the excel
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')









