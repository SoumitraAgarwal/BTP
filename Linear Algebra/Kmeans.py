import math
import pickle
import random
import numpy as np
import pandas as pd
import time
import os

from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.optimize import minimize
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
plt.switch_backend('agg')

random.seed(311311)

def cluster(data):
	kmeans 		= KMeans(n_clusters=2, random_state=0).fit(data)
	centres 	= kmeans.cluster_centers_
	idx 		= np.asarray(kmeans.labels_, dtype = bool)
	randomTrain = data[idx]
	randomTar 	= target[idx]
	regleft		= linear_model.LinearRegression()
	regleft.fit(randomTrain, randomTar)
	predl 		= regleft.predict(randomTrain)
	x11, x12 	= randomTrain.T

	fig 	= plt.figure()
	ax1 	= fig.add_subplot(111)
	x1,x2 	= randomTrain.T

	ax1.scatter( x1,x2, s = 1, marker="o")

	idx 		= ~idx
	randomTrain = data[idx]
	randomTar 	= target[idx]
	regright	= linear_model.LinearRegression()
	regright.fit(randomTrain, randomTar)
	predr 		= regright.predict(randomTrain)

	x1,x2 		= randomTrain.T
	x21, x22 	= randomTrain.T
	ax1.scatter( x1,x2, s = 1,  marker="s")

	plt.savefig('Plots/Kmeans' + str(presentNode) + '.png')
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
entrythresh = 30


while(len(queue)>0):
	presentNode = queue[0]

	print("Training for node " + str(presentNode))
	data 	= pd.read_csv("Data/" + str(presentNode) + '.csv')
	target 	= data[['Y']]
	data 	= data[['X1', 'X2']]
	data 	= data.values
	target 	= target.values
	entries 	= len(data)
	model	= linear_model.LinearRegression()
	model.fit(data, target)
	pickle.dump(model, open("Model/Rho_" + str(presentNode) + '.p', "wb" ))


	featureSize	 = len(data[0])


	idx,regleft, regright, centres 	= cluster(data)
	centres = centres.tolist()
	
	def loss_function(W):
		# print(W)
		
		Gamma = W[:-2]	
		alpha = W[-2]
		beta  = W[-1]

		# print(Gamma.shape)
		# print(data.shape)
		error 		= 0
		Delta 		= []
		sumGamma 	= sum(Gamma)
		modGamma 	= np.dot(Gamma, Gamma)
		for game in Gamma:
			del1 = 1.0*(2*alpha*alpha*modGamma 	+ game*sumGamma*(beta - alpha)*(beta + alpha))/ (modGamma*alpha*(alpha*alpha + beta*beta)*2)
			del2 = 1.0*(2*beta*beta*modGamma 	- game*sumGamma*(beta - alpha)*(beta + alpha))/ (modGamma*beta*(alpha*alpha + beta*beta)*2)
			Delta.append([max(del1, 0), max(del2,0)])
		
		Delta = np.asarray(Delta)
		for i in range(len(data)):
			error += 0.5*Delta[i][0]*(target[i] - regleft.predict([data[i]]))*(target[i] - regleft.predict([data[i]])) + 0.5*Delta[i][1]*(target[i] - regright.predict([data[i]]))*(target[i] - regright.predict([data[i]]))
		return error
	

	flag1 = 0
	flag2 = 0
	for i in range(iterations):

		print("Iteration : " + str(i))

		W 		= np.asarray(np.random.uniform(0,1,entrythresh + 2))
		start = 0
		for picker in range(5):
			# print('Batch : ' + str(picker))
			data 	= pd.read_csv("Data/" + str(presentNode) + '.csv')
			target 	= data[['Y']]
			data 	= data[['X1', 'X2']]
			data 	= data.values
			data 	= data[start:start + entrythresh]
			target 	= target.values
			target 	= target[start:start + entrythresh]
			W 		= minimize(loss_function, W, options={'maxiter':2}).x
			print("Error : " + str(loss_function(W)))

			if(start + 2*entrythresh > entries):
				start = 0

	Gamma = W[:-2]	
	alpha = W[-2]
	beta  = W[-1]
	error 	= 0
	Delta 		= []
	sumGamma 	= sum(Gamma)
	modGamma 	= np.dot(Gamma, Gamma)
	for game in Gamma:
		del1 = 1.0*(2*alpha*alpha*modGamma 	+ game*sumGamma*(beta - alpha)*(beta + alpha))/ (modGamma*alpha*(alpha*alpha + beta*beta)*2)
		del2 = 1.0*(2*beta*beta*modGamma 	- game*sumGamma*(beta - alpha)*(beta + alpha))/ (modGamma*beta*(alpha*alpha + beta*beta)*2)
		Delta.append([max(del1, 0), max(del2,0)])
	
	Delta = np.asarray(Delta)

	p1 = []
	p2 = []

	for i in range(len(Delta)):
		p1.append(1.0*Delta[i][0]*alpha)
		p2.append(1.0*Delta[i][1]*beta)

	p1 = np.asarray(p1)
	p2 = np.asarray(p2)
	P 	= np.asarray([p1, p2]).T
	Mat = np.linalg.lstsq(np.vstack([data.T, np.ones(len(data))]).T, P)[0]
	x1, x2 = Mat.T
	
	w1 = x1[:-1]
	b1 = x1[-1]
	w2 = x2[:-1]
	b2 = x2[-1]

	
	idx = []

	data 	= pd.read_csv("Data/" + str(presentNode) + '.csv')
	target 	= data[['Y']]
	data 	= data[['X1', 'X2']]
	data 	= data.values
	target 	= target.values
	for dat in data:

		u1 = np.dot(w1, dat) + b1
		u2 = np.dot(w2, dat) + b2

		if(u1 > u2):
			idx.append(True)
		else:
			idx.append(False)	

	
	idx	= np.asarray(idx)
	W = w1.tolist() + [b1] + w2.tolist() + [b2]
	W = np.asarray(W)
	
	print(sum(idx), sum(~idx))		


	if(sum(idx) == 0 or sum(~idx) == 0):
		flag1 = 0
		flag2 = 0
		pickle.dump(W, open("Model/W_" + str(presentNode) + '.p', "wb" ))
		queue = queue[1:]
		continue

	if(sum(idx)> entrythresh):
		flag1 = 1
	else:
		flag1 = 0

	if(sum(~idx)> entrythresh):
		flag2 = 1
	else:
		flag2 = 0
		
	
	randomTrain = data[idx]
	randomTar 	= target[idx]
	regleft		= linear_model.LinearRegression()
	regleft.fit(randomTrain, randomTar)
	
	fig 	= plt.figure()
	ax1 	= fig.add_subplot(111)

	x1,x2 	= randomTrain.T
	
	result = pd.DataFrame(data = {'X1':x1,'X2':x2,'Y' :randomTar.ravel()})
	result.to_csv("Data/"+ str(2*presentNode + 1) + '.csv', index = False)
	ax1.scatter( x1,x2, s = 1, marker="o")

	idx 		= ~idx
	randomTrain = data[idx]
	randomTar 	= target[idx]
	x1,x2 		= randomTrain.T
	result 		= pd.DataFrame(data = {'X1':x1,'X2':x2,'Y' :randomTar.ravel()})
	result.to_csv("Data/"+ str(2*presentNode + 2) + '.csv', index = False)
	x1,x2 		= randomTrain.T
	ax1.scatter( x1,x2, s = 1,  marker="s")
	plt.savefig('Plots/Node' + str(presentNode) + '.png')

	regright	= linear_model.LinearRegression()
	regright.fit(randomTrain, randomTar)

	if(flag1 == 1):
		queue.append(2*presentNode + 1)
	if(flag2 == 1):
		queue.append(2*presentNode + 2)

	print(queue)
	print(W)
	pickle.dump(W, open("Model/W_" + str(presentNode) + '.p', "wb" ))
	queue = queue[1:]
