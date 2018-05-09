import math
import pickle
import numpy as np
import pandas as pd
import time

from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.optimize import minimize
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
plt.switch_backend('agg')


def cluster(data):
	kmeans 		= KMeans(n_clusters=2, random_state=0).fit(data)
	centres 	= kmeans.cluster_centers_
	idx 		= np.asarray(kmeans.labels_, dtype = bool)
	randomTrain = data[idx]
	randomTar 	= target[idx]
	regleft		= linear_model.LinearRegression()
	regleft.fit(randomTrain, randomTar)
	predl 		= regleft.predict(randomTrain)
	# x11, x12 	= randomTrain.T

	# fig 	= plt.figure()
	# ax1 	= fig.add_subplot(111)
	# x1,x2 	= randomTrain.T

	# ax1.scatter( x1,x2, s = 1, marker="o")

	idx 		= ~idx
	randomTrain = data[idx]
	randomTar 	= target[idx]
	regright	= linear_model.LinearRegression()
	regright.fit(randomTrain, randomTar)
	predr 		= regright.predict(randomTrain)

	# x1,x2 		= randomTrain.T
	# x21, x22 	= randomTrain.T
	# ax1.scatter( x1,x2, s = 1,  marker="s")

	# plt.savefig('Plots/Kmeans' + str(presentNode) + '.png')
	# fig 	= plt.figure()
	# ax1 	= fig.add_subplot(111, projection = '3d')

	# x1, x2 = data.T

	# ax1.scatter(x1, x2, target)
	# colors 	= cm.rainbow(np.linspace(0, 1, 2))
	# ax1.scatter(x11, x12, predl, s= 1, color=colors[0])
	# ax1.scatter(x21, x22, predr, s= 1, color=colors[1])
	# plt.show()

	return idx, regleft, regright, centres

queue 		= []
queue.append(0)
iterations 	= 3
entrythresh = 50

columns = ['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10','X_11','X_12','X_13','X_14','X_15','X_16','X_17','X_18','Y1']

while(len(queue)>0):
	presentNode = queue[0]

	print("Training for node " + str(presentNode))
	data 	= pd.read_csv("Data/" + str(presentNode) + '.csv')
	target 	= data[[columns[-1]]]
	data 	= data[columns[:-1]]
	data 	= data.values
	target 	= target.values
	
	model	= linear_model.LinearRegression()
	model.fit(data, target)
	pickle.dump(model, open("Model/Rho_" + str(presentNode) + '.p', "wb" ))


	featureSize	 = len(data[10])


	idx,regleft, regright, centres 	= cluster(data)
	centres = centres.tolist()
	W 		= centres[0] + [0] + centres[1] + [0]
	W 		= np.asarray(W)
	

	def loss_function(W):
		w1 = W[:featureSize]
		b1 = W[featureSize + 1]	
		w2 = W[featureSize + 1: -1]
		b2 = W[-1]

		leftprobs 	= []
		rightprobs 	= []
		error 		= 0

		for dat in data:
			u1 = 1.0*(np.dot(w1, dat) + b1)/(np.dot(w1, w1) + b1*b1)
			u2 = 1.0*(np.dot(w2, dat) + b2)/(np.dot(w2, w2) + b2*b2)

			p1 = 1.0*math.exp(u1)/(math.exp(u1) + math.exp(u2))	
			p2 = 1.0 - p1	
			# print(p1, p2)
			
			leftprobs.append(p1)
			rightprobs.append(p2)

		sumleft = sum(leftprobs)
		sumrigh	= sum(rightprobs)

		for i in range(len(leftprobs)):
			error += 0.5*leftprobs[i]*(target[i] - regleft.predict([data[i]]))*(target[i] - regleft.predict([data[i]]))/sumleft + 0.5*rightprobs[i]*(target[i] - regright.predict([data[i]]))*(target[i] - regright.predict([data[i]]))/sumrigh
		
		return error
	

	flag1 = 0
	flag2 = 0
	for i in range(iterations):

		print("Iteration : " + str(i))
		W 			= minimize(loss_function, W, options={'maxiter':2}).x
		print("Error : " + str(loss_function(W)))
		idx = []
		for dat in data:

			w1 = W[:featureSize]
			b1 = W[featureSize + 1]	
			w2 = W[featureSize + 1: -1]
			b2 = W[-1]

			u1 = 1.0*(np.dot(w1, dat) + b1)/(np.dot(w1, w1) + b1*b1)
			u2 = 1.0*(np.dot(w2, dat) + b2)/(np.dot(w2, w2) + b2*b2)

			p1 = 1.0*math.exp(u1)/(math.exp(u1) + math.exp(u2))
			if(p1 > 0.5):
				idx.append(True)
			else:
				idx.append(False)	

		idx 		= np.asarray(idx)
		
		if(sum(idx) == 0 or sum(~idx) == 0):
			flag1 = 0
			flag2 = 0
			break

		if(sum(idx)> entrythresh):
			flag1 = 1
		else:
			flag1 = 0

		if(sum(~idx)> entrythresh):
			flag2 = 1
		else:
			flag2 = 0
			
		if(flag1*flag2 == 0):
			break
		

		randomTrain = data[idx]
		randomTar 	= target[idx]
		regleft		= linear_model.LinearRegression()
		regleft.fit(randomTrain, randomTar)
		
		# fig 	= plt.figure()
		# ax1 	= fig.add_subplot(111)

		print(randomTrain.shape)
		print(randomTar.shape)
		complete = np.hstack([randomTrain, randomTar])
		
		result = pd.DataFrame(complete, columns = columns)
		result.to_csv("Data/"+ str(2*presentNode + 1) + '.csv', index = False)
		# ax1.scatter( x1,x2, s = 1, marker="o")

		idx 		= ~idx
		randomTrain = data[idx]
		randomTar 	= target[idx]
		print(randomTrain.shape)
		print(randomTar.shape)
		complete 	= np.hstack([randomTrain, randomTar])
		
		result = pd.DataFrame(complete, columns = columns)
		result.to_csv("Data/"+ str(2*presentNode + 2) + '.csv', index = False)
		# x1,x2 		= randomTrain.T
		# ax1.scatter( x1,x2, s = 1,  marker="s")
		# plt.savefig('Plots/Node' + str(presentNode) + '.png')

		regright	= linear_model.LinearRegression()
		regright.fit(randomTrain, randomTar)

		
	if(flag1 == 1):
		queue.append(2*presentNode + 1)
	if(flag2 == 1):
		queue.append(2*presentNode + 2)

	pickle.dump(W, open("Model/W_" + str(presentNode) + '.p', "wb" ))
	queue = queue[1:]
