# -*- coding: utf-8 -*-
"""
Chapter 6 - Biophysical modeling.

Created on Dec 21 2018

@author: ethan mcbride
"""

#**************************
# This uses object-oriented programming to create multiple biophysical models of neurons
# 
#
#************************


# %% Init
import numpy as np
import matplotlib.pyplot as plt


# Define classes
class Neuron():
    def __init__(self):
        self.C = 0.281
        self.gL = 0.030
        self.vR = -60.6
        self.vT = -50.4

    def create_injection_current(self):
        self.currentInj = np.append(np.zeros(10),np.arange(100)/100.)
        self.T = len(self.currentInj)
    
    def leaky_integrate_and_fire(self):
        self.timeseries = np.linspace(0,self.T-1,self.T)
        self.V = np.ones(self.T)*self.vR
        ii=0 #initialize the index counter
        while ii < self.T-2:
            dV = (-self.gL*(self.V[ii] - self.vR)+self.currentInj[ii])/self.C
            self.V[ii+1]=self.V[ii]+dV
            if self.V[ii+1]>=self.vT:
                self.V[ii+1]=20
                self.V[ii+2]=self.vR
                ii+=1 #increment
            ii+=1

    def plot_neuron(self):
        fig = plt.figure(figsize=(5,5))
        
        ax = fig.add_subplot(211)
        ax.plot(self.timeseries,self.currentInj,c='k')
        ax.set_title('current injection',style='italic')
        ax.set_ylabel('current (nA)',style='italic')
        
        ax2 = fig.add_subplot(212)
        ax2.plot(self.timeseries,self.V,c='k')
        ax2.set_title('integrate and fire voltage response',style='italic')
        ax2.set_xlabel('time (ms)',style='italic')
        ax2.set_ylabel('voltage (mV)',style='italic')
        plt.tight_layout()
        #plt.show()
        plt.savefig('Integrate and fire voltage response.png')


class Neuron2():
    def __init__(self):
        self.C = 0.281
        self.gL = 0.030
        self.vR = -60.6
        self.vT = -50.4
    
    def create_injection_current(self,mag=1):
        self.currentInj = np.arange(100)/100.*mag
        self.T = len(self.currentInj)
        
    def leaky_integrate_and_fire(self):
        self.timeseries = np.linspace(0,self.T-1,self.T)
        self.V = np.ones(self.T)*self.vR
        ii=0 # init counter
        self.spikeCounter=0
        while ii < self.T-2:
            dV = (-self.gL*(self.V[ii] - self.vR)+self.currentInj[ii])/self.C
            self.V[ii+1]=self.V[ii]+dV
            if self.V[ii+1]>self.vT:
                self.V[ii+1]=20
                self.V[ii+2]=self.vR
                ii+=1
                self.spikeCounter+=1
            ii+=1
            
def plotFI(currentMags,spikes):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(currentMags,spikes,c='k',edgecolor='w',s=50)
    ax.set_xlabel('current injection maximum (nA)',style='italic')
    ax.set_ylabel('number of spikes',style='italic')
    ax.set_title('Firing as function of current injected',style='italic')
    #plt.show()
    plt.savefig('Firing as function of current injected.png')
    

            
# %% test the model
        
myFirstNeuron = Neuron()
myFirstNeuron.create_injection_current()
myFirstNeuron.leaky_integrate_and_fire()
myFirstNeuron.plot_neuron()


# %% plot # spikes as function of current injection

spikes=[]
currentMags=np.arange(0.1,10,0.1)
for mag in currentMags:
    mySecondNeuron = Neuron2()
    mySecondNeuron.create_injection_current(mag)
    mySecondNeuron.leaky_integrate_and_fire()
    spikes.append(mySecondNeuron.spikeCounter)
plotFI(currentMags,spikes)
