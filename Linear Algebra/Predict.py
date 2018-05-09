import os
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
plt.switch_backend('agg')

data = pd.read_csv('predict.csv')

target 	= data[['Y']]
data 	= data[['X1', 'X2']]
data 	= data.values
target 	= target.values
predict = []
featureSize = len(data[0])
nodes = os.listdir('Model')
print(nodes)
itera = 0
for dat in data:
	itera += 1
	print(itera)
	Node = 0
	while(True):
		print("Node :" + str(Node))
		W = pickle.load(open('Model/W_' + str(Node) + '.p', 'rb'))
		w1 = W[:featureSize]
		b1 = W[featureSize + 1]	
		w2 = W[featureSize + 1: -1]
		b2 = W[-1]


		u1 = np.dot(w1, dat) + b1
		u2 = np.dot(w2, dat) + b2

		if('W_' + str(2*Node + 1) + '.p' not in nodes):
			break

		else:	
			if(u1 > u2):
				Node = 2*Node + 1

			else:
				Node = 2*Node + 2


	Model = pickle.load(open('Model/Rho_' + str(Node) + '.p', 'rb'))

	predict.append(Model.predict([dat]))


fig 	= plt.figure()
ax1 	= fig.add_subplot(111, projection = '3d')

x1, x2 = data.T

predict = np.asarray(predict)
ax1.scatter(x1, x2, target)
colors 	= cm.rainbow(np.linspace(0, 1, 2))
ax1.scatter(x1, x2, predict, s= 1, color=colors[0], label =  'Predicted')
ax1.scatter(x1, x2, target, s= 1, color=colors[1], label = 'Target')
plt.legend(loc='upper left');
plt.savefig('Plots/Prediction.png')

print('Accuracy ' + str(r2_score(target.ravel(), predict.ravel())))