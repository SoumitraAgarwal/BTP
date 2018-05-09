import math
import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.optimize import minimize
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
plt.switch_backend('agg')

node 	= 0

data 	= pd.read_csv('Data/' + str(node) + '.csv') 
target 	= data[['Y']]
data 	= data[['X1', 'X2']]
data 	= data.values
target 	= target.values
	
W 		= pickle.load(open('Model/W_' + str(node) + '.p', 'rb'))

featureSize = 2

idx 	= []
for dat in data:

	w1 = W[:featureSize]
	b1 = W[featureSize + 1]	
	w2 = W[featureSize + 1: -1]
	b2 = W[-1]

	u1 = np.dot(w1, dat) + b1
	u2 = np.dot(w2, dat) + b2

	p1 = 1.0*math.exp(u1)/(math.exp(u1) + math.exp(u2) + 0.0000001)
	if(p1 > 0.5):
		idx.append(True)
	else:
		idx.append(False)	

fig 		= plt.figure()
ax1 		= fig.add_subplot(111)
idx 		= np.asarray(idx)

randomTrain = data[idx]
randomTar 	= target[idx]
x1,x2 		= randomTrain.T
ax1.scatter( x1,x2, s = 1, marker="o")

idx 		= ~idx
randomTrain = data[idx]
randomTar 	= target[idx]
x1,x2 		= randomTrain.T
ax1.scatter( x1,x2, s = 1,  marker="s")
plt.savefig('Plots/Node' + str(node) + '.png')