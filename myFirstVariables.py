# coding: utf-8
# Chapter 2: Basics
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])

#column vector
B = np.array([3],[5],[4],[4],[2])
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])

#column vector
B = np.array([[3],[5],[4],[4],[2]])
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])

#column vector
B = np.array([[3],[5],[4],[4],[2]])

A_dim = np.shape(A)
print('A dimensions',A_dim)
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])
print('A=',A)

#column vector
B = np.array([[3],[5],[4],[4],[2]])
print('B=',B)

A_dim = np.shape(A)
print('A dimensions',A_dim)

B_dim = np.shape(B)
print('B dimensions',B_dim)
# vectors - rows v columns
import numpy as np

# row vector
A = np.array([1,2,3,4,2])
print('A=',A)

#column vector
B = np.array([[3],[5],[4],[4],[2]])
print('B=',B)

A_dim = np.shape(A)
print('A dimensions',A_dim)

B_dim = np.shape(B)
print('B dimensions',B_dim)

B.T
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

A[2,:]
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[2,:]
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or use 0 to mean/sum across all elements

print('dim 1 sum ',sum(C,1))
print('dim 2 sum ',sum(C,2))
print('dim 0 sum ',sum(C,0))
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or use 0 to mean/sum across all elements

print('dim 1 sum ',sum(C,1))
print('dim 2 sum ',sum(C,2))
print('dim 0 sum ',mean(C,0))
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or use 0 to mean/sum across all elements

print('dim 1 sum ',np.sum(C,1))
print('dim 2 sum ',np.sum(C,2))
print('dim 0 sum ',np.sum(C,0))
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or use 0 to mean/sum across all elements

print('dim 1 sum ',np.mean(C,1))
print('dim 2 sum ',np.mean(C,2))
print('dim 0 sum ',np.mean(C,0))
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or don't to mean/sum across all elements

print('dim 1 mean ',np.mean(C,0))
print('dim 2 mean ',np.mean(C,1))
print('dim 0 mean ',np.mean(C))
# matrices are a little more of a pain in python

C = np.array([[1,2,3],[4,5,6],[7,8,9]])

#"shape" is the dimensions
sh=np.shape(C)
print('shape is ', sh)

#"size" is the number of elements
si=np.size(C)
print('size is', si)

C[:2,1:]

#when doing means or sums on matrices, specify the dimension, or don't to mean/sum across all elements
print('C=',C)
print('dim 1 mean ',np.mean(C,0))
print('dim 2 mean ',np.mean(C,1))
print('dim 0 mean ',np.mean(C))
# practice dealing with spike rates

neuralFiringRates = np.array([0.5, 1, 2, 4, 8, 0.25, 3])
numberOfNeurons = np.array([45, 15, 35, 20, 50, 36, 101])

#matrix multiplication or in this case dot product (".inner") has to be explicitly declared
expectedSpikeCount = np.inner(neuralFiringRates,numberOfNeurons)
print(expectedSpikeCount)
# practice dealing with spike rates

neuralFiringRates = np.array([0.5, 1, 2, 4, 8, 0.25, 3])
numberOfNeurons = np.array([45, 15, 35, 20, 50, 36, 101])

#matrix multiplication or in this case dot product (".inner") has to be explicitly declared
expectedSpikeCount = np.inner(neuralFiringRates,numberOfNeurons)
print('dot product:',expectedSpikeCount)

#element-wise multiplication is implied with *
expectedSpikeCount = neuralFiringRates*numberOfNeurons
print('element-wise product:',expectedSpikeCount)
# the "where" function

indices = np.where(neuralFiringRates>3)
# the "where" function

indices = np.where(neuralFiringRates>3)

indices
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))

RTmean = np.nanmean(RTs)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))

RTmean = np.nanmean(RTs)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print('RTs')

RTmean = np.nanmean(RTs)
print('RTmean')
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print(RTs)

RTmean = np.nanmean(RTs)
print(RTmean)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print(RTs)

RTmean = np.nanmean(RTs)
print(RTmean,0)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print(RTs)

RTmean = np.nanmean(RTs,0)
print(RTmean)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print(RTs)

RTmean = np.nanmean(RTs,0)
print(RTmean)

#or delete the columns with nan
r,c=np.where(np.isnan(RTs)==1)
RTs=np.delete(RTs,c,r)
print(RTs)
# missing data and NaN's

rtP1 = np.array([1000,1500,500,1200])
rtP2 = np.array([2000,1700,3000,float('NaN')])

RTs = np.vstack((rtP1,rtP2))
print(RTs)

RTmean = np.nanmean(RTs,0)
print(RTmean)

#or delete the columns with nan
r,c=np.where(np.isnan(RTs)==1)
print('row:',r,'column:',c)
RTs=np.delete(RTs,c,r)
print(RTs)
import matplotlib.pyplot as plt
# woo graphs

sampleSize = 10
meanIQ = 100
stdIQ = 100
sampleIQ = np.random.randn(sampleSize)*stdIQ+meanIQ
f = plt.figure()
ax = plt.hist(sampleIQ,bins=20)
plt.show()
import matplotlib.pyplot as plt
# woo graphs

sampleSize = 10
meanIQ = 100
stdIQ = 100
sampleIQ = np.random.randn(sampleSize)*stdIQ+meanIQ
f = plt.figure()
ax = plt.hist(sampleIQ,bins=20)
plt.show()
import matplotlib.pyplot as plt
# woo graphs

sampleSize = 10000
meanIQ = 100
stdIQ = 100
sampleIQ = np.random.randn(sampleSize)*stdIQ+meanIQ
f = plt.figure()
ax = plt.hist(sampleIQ,bins=20)
plt.show()
# practice dealing with spike rates

neuralFiringRates = np.array([0.5, 1, 2, 4, 8, 0.25, 3])
numberOfNeurons = np.array([45, 15, 35, 20, 50, 36, 101])

#matrix multiplication or in this case dot product (".inner") has to be explicitly declared
expectedSpikeCount = np.inner(neuralFiringRates,numberOfNeurons)
print('dot product:',expectedSpikeCount)

#element-wise multiplication is implied with *
expectedSpikeCount = neuralFiringRates*numberOfNeurons
print('element-wise product:',expectedSpikeCount)

#outer product, or tensor product, gives you all the possible multiplications. 
#can be useful for looking at all combinations, like combinations of different conditions

exposureDurations = np.array([1,2,3,4,5])
lightLevels = np.array([1,2,4,8,16])
photonCounts = np.outer(exposureDurations,lightlevels)
photonCounts
# practice dealing with spike rates

neuralFiringRates = np.array([0.5, 1, 2, 4, 8, 0.25, 3])
numberOfNeurons = np.array([45, 15, 35, 20, 50, 36, 101])

#matrix multiplication or in this case dot product (".inner") has to be explicitly declared
expectedSpikeCount = np.inner(neuralFiringRates,numberOfNeurons)
print('dot product:',expectedSpikeCount)

#element-wise multiplication is implied with *
expectedSpikeCount = neuralFiringRates*numberOfNeurons
print('element-wise product:',expectedSpikeCount)

#outer product, or tensor product, gives you all the possible multiplications. 
#can be useful for looking at all combinations, like combinations of different conditions

exposureDurations = np.array([1,2,3,4,5])
lightLevels = np.array([1,2,4,8,16])
photonCounts = np.outer(exposureDurations,lightLevels)
photonCounts
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
photonCountsVector
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
firingRates = photonCountsVector*np.round(5*np.random.rand(len(photonCountsVector)),1)
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
firingRates = photonCountsVector*np.round(5*np.random.rand(len(photonCountsVector)),1)
firingRates
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
firingRates = photonCountsVector*np.round(5*np.random.rand(len(photonCountsVector)),1)
f = plt.figure()
ax = plt.plot(firingRates,lw=3) #lw: linewidth = 3
plt.xlabel('photon count')
plt.ylabel('spike count')
plt.show()
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
firingRates = photonCountsVector*np.round(5*np.random.rand(len(photonCountsVector)),1)
f = plt.figure()
ax = plt.plot(firingRates,lw=3) #lw: linewidth = 3
plt.xlabel('photon count')
plt.ylabel('spike count')
plt.show()
# plot fake spike counts

photonCountsVector = np.unique(photonCounts.flatten())
firingRates = photonCountsVector*np.round(5*np.random.rand(len(photonCountsVector)),1)
f = plt.figure()
ax = plt.plot(firingRates,lw=3) #lw: linewidth = 3
plt.xlabel('photon count')
plt.ylabel('spike count')
plt.show()
# saving variables

get_ipython().run_line_magic('save', 'myFirstVariables')
# saving variables

get_ipython().run_line_magic('save', 'myFirstVariables 1-50')
