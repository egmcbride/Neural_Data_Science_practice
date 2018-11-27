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
numOri = len(ori)
trialIndices = defaultdict(list)

# %% 2. & 3. Pruning and Formatting

# Step 1: Identify trial numbers (indices) that correspond to a given orientation

for tt in range(numTrials):
    for oo in ori:
        if allOris[tt]==oo:
            trialIndices[oo].append(tt)
            
# Step 2: sort spike times into corresponding conditions

numOris = len(ori)
allOris=[matIn['DATA'][_][0][1][0][0] for _ in range(numTrials)]
for orind.oo in enumerate(ori):
    for tt in trialIndices[oo]:
        for cc in range(numChannels):
            spikeTimes[tt][orind][cc] = [spike[2] for spike in matIn['DATA'][tt][0][0] 
            if (spike[0]==cc) and (spike[1] not in noiseCodes)]
            
            
            
