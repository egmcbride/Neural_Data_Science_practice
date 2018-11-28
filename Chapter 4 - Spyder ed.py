# -*- coding: utf-8 -*-
"""
Chapter 4 - Spyder ed.

Created on Tue Nov 27 09:08:24 2018

@author: ethan_000
"""

#**************************
# This program recreates the spike count correlation analysis from Snyder et al. (2015)
# It assumes that the sata file from the recording array exists in the same 
# directory as the analysis file.
# Created (really, transcribed) by:
# Ethan McBride
# ethan.g.mcbride@gmail.com
# versions
#
#
#************************


# %% 0. Python Init
# for Python, we load the functions that we'll be using and assign some of them functions

import scipy.io
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc

numChannels = 96
noiseCodes = [0,255]
timeBase = np.arange(-.2,2.5,.01)
gratingOn = 0
gratingOff = 2

# %% 1. Loader. Load the data file

# had to add an r to make it a "raw string" for some reason
matIn = scipy.io.loadmat(r"C:\Users\ethan_000\Documents\GitHub\Neural_Data_Science_practice\arrayDATA.mat")

numTrials = len(matIn['DATA'])
allOris=[matIn['DATA'][_][0][1][0][0] for _ in range(numTrials)]
ori = list(set(allOris))
numOris = len(ori)
trialIndices = defaultdict(list)

# %% 2. & 3. Pruning and Formatting

# Step 1: Identify trial numbers (indices) that correspond to a given orientation

for tt in range(numTrials):
    for oo in ori:
        if allOris[tt]==oo:
            trialIndices[oo].append(tt)

# Step 2: find spike times for all combinations of conditions

# use a dict, which allows dynamic adding of key:value pairs
# lambda allows another layer of dicts 
linearizedSpikeTimes = defaultdict(lambda: defaultdict(list))
for eachtrial in matIn["DATA"]:
    stimori = eachtrial[0][1][0][0]
    for eachspike in eachtrial[0][0]:
        if eachspike[1] not in [0,255]: #not in noiseCodes
            trode = eachspike[0]
            spikeTimes = eachspike[2]
            linearizedSpikeTimes[trode][stimori].append(spikeTimes)

# %% 4a. calculate PSTHs
   
#makes histograms of spike rates in each condition
PSTHs = defaultdict(lambda: defaultdict(list))
for unitkey in linearizedSpikeTimes.keys():
    PSTHs[unitkey][0],bins = np.histogram(linearizedSpikeTimes[unitkey][0], bins=timeBase)
    PSTHs[unitkey][90],bins = np.histogram(linearizedSpikeTimes[unitkey][90], bins=timeBase)
    
    
# %% 5a. data browser 

for unitkey in PSTHs.keys():
    fig=plt.figure(facecolor='w')
    for orind,oo in enumerate(ori):
        ax = fig.add_subplot(2,1,orind+1)
        ax.plot(timeBase[:-1],PSTHs[unitkey][oo],lw=3,color='b')
        ax.set_xlim([-.2,2.5])
        ax.vlines(gratingOn,0,max(PSTHs[unitkey][oo]),color='k',linestyle='--')
        ax.vlines(gratingOff,0,max(PSTHs[unitkey][oo]),color='k',linestyle='--')
        ax.set_ylabel('spike count per bin')
        ax.set_xlabel('time in seconds')
        ax.set_title('Channel = '+str(int(unitkey))+' Orientation = '+str(oo))
    plt.tight_layout()
    plt.show()
    
    

# %% Spike count correlations!
    
# 1. remove bad channels
    
livePSTHs = PSTHs #make a copy to make sure we keep our place
#deadchannels = [21, 31, 34, 56, 68] #real bad channels
deadchannels = [21,31,34,56,68, 7,16,32,37,39,41,43,45,47,48,51,52,54,94,95] #dead and noisy channels
[livePSTHs.pop(dead, None) for dead in deadchannels ]
numChanLeft = len(livePSTHs)
         

# 2. Calculating the correlations

rSC = np.zeros(shape=(len(livePSTHs),len(livePSTHs), len(ori))) #preallocate
for rowind,rowkey in enumerate(livePSTHs.keys()): #loop over rows (units)
    for colind,colkey in enumerate(livePSTHs.keys()): #loop over columns (same units)
        for orind,oo in enumerate(ori): #loop over orientations
            rSC[rowind,colind,orind],dummy = sc.pearsonr(livePSTHs[rowkey][oo][11:211], 
               livePSTHs[colkey][oo][11:211]) #noise correlation ONLY OVER STIM (11:211))
globalMean = np.mean(rSC)


# %% Plot correlations

#important for resizing colorbar!!
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.set_cmap('jet')

fig=plt.figure(facecolor='w')
for orind,oo in enumerate(ori):
    ax = fig.add_subplot(1,numOris,orind+1)
    img=ax.imshow(rSC[:,:,orind])
#    ax.axis('equal')
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(img, cax=cax)

    ax.set_title('Orientation: '+str(oo)+'\xb0')
plt.tight_layout()
plt.show()


""" 
Keep in mind convolution!
Convolution is just a moving average
Most often done on bin widths of 1ms with a gaussian kernel

Technically it is:
    multiplication of each value in a time series with a corresponding value
    in a "kernel" that slides across the time series, then summing up all of
    these products to yield a new value (often divided by the sum of all 
    numbers in the kernel to get an average, instead of a sum)
    
    Convolution is "zero-padded", so that values at the edges are multiplied by zero and fall off
    This results in a n+(k-1) elements output
    "valid" results are shorter, n-(k-1)
    use odd-numbered kernels, so that they do not phase shift (even kernels phase shift)
    
