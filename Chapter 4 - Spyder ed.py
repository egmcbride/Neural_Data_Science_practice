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
    
    

# %% SPike count correlations!
    
         
