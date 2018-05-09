import random
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.switch_backend('agg')
random.seed(311)


def generate(radius, centre):
	alpha 		= 2 * math.pi * random.random()
	r 			= radius*random.random()
	x 			= r*math.cos(alpha) + centre[0]
	y 			= r*math.sin(alpha) + centre[1]

	return [x,y]

k = 8
n = 500
ranger = 500

C = []
X = []
Y = []
for j in range(k):
	
	T 		= [random.uniform(0, ranger), random.uniform(0, ranger)]	
	temp 	= []
	C.append([[j*ranger + random.uniform(0, ranger), ranger*random.uniform(0, k)], 400*random.uniform(0, 1)])
	for i in range(n):
		temp.append(generate(C[j][1], C[j][0]))
	
	temp = np.asarray(temp)
	Y.append(np.matmul(temp,T))
	X.append(temp)

X = np.asarray(X)
Y = np.asarray(Y)

fig 	= plt.figure()
ax1 	= fig.add_subplot(111, projection = '3d')
colors 	= cm.rainbow(np.linspace(0, 1, len(Y)))

for i in range(k):
	x1, y1 = X[i].T

	ax1.scatter( x1,y1, Y[i], s = 3, marker="o", label='target', color=colors[i])

plt.savefig('Data.png')

X1 = []
X2 = []
for i in range(k):
	x1,x2 = X[i].T
	X1.append(x1)
	X2.append(x2)

X1 	= np.asarray(X1)
X2 	= np.asarray(X2)
Y 	= Y.ravel()
X1 	= X1.ravel()
X2 	= X2.ravel()
X1 = preprocessing.scale(X1)
X2 = preprocessing.scale(X2)
Y  = preprocessing.scale(Y)

data = pd.DataFrame(data = {
	'X1':X1,
	'X2':X2,
	'Y' :Y
	})

data 	= data.sample(frac=1).reset_index(drop=True)
data.to_csv('data.csv', index = False)