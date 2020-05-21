import sys
import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import random 
from sklearn import metrics
import glob 
import cv2 
from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
import numpy as np


def Kmean(X, k, maxiter): 
	C = [X[int(random.random()*len(X))] for i in range(k)]
	labels = []

	for i in range(maxiter): 
		labels = []
		for j in range(len(X)):
			temp = []
			for m in range(k):
				temp.append(np.dot(X[j]-C[m], X[j]-C[m])**2)
			
			labels.append(np.argmin(np.array(temp)))
		C = []
		for j in range(k):
			temp1 = []
			for m in range(len(X)):
				if(labels[m]==j):
					temp1.append(X[m])

			if(len(temp1)==0):
				C.append(X[int(random.random()*len(X))])
			else:	
				mean = np.mean(temp1, axis = 0)
				flag = []
				for n in range(len(temp1)):
					flag.append(np.dot(temp1[n]-mean, temp1[n]-mean)**2)
				C.append(temp1[np.argmin(flag)]) 

	return C, labels


mmmm = 0

for filename in glob.glob('Q1-images/*'): 
	X1 = []
	X2 = []

	I = cv2.imread(filename)
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			X1.append(I[i][j][:].tolist())
			X2.append(I[i][j][:].tolist()+[i,j])


	X1 = np.array(X1)

	C, labels = Kmean(X1, 7, 10)

	print X1.shape 
	print X1[:,0].shape 
	print np.array(labels).shape  
	print X1[:,1].shape 

	count = 0
	I_out = np.zeros(I.shape)
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			
			I_out[i][j][:] = C[labels[count]]
			count = count + 1

	f = 'Output/Q1/' + str(mmmm) + '_7.jpg'
	cv2.imwrite(f, I_out)
	
	print filename[-15:]

	if(filename[-15:]=='2or4objects.jpg'):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		

		ax.scatter(X1[:,0], X1[:,1], X1[:,2], c=np.array(labels), cmap=plt.cm.Paired)

		ax.set_xlabel('R')
		ax.set_ylabel('G')
		ax.set_zlabel('B')

		plt.show()


	X2 = np.array(X2)

	C, labels = Kmean(X2, 10, 10)


	count = 0
	I_out = np.zeros(I.shape)
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			
			I_out[i][j][:] = C[labels[count]][:3]
			count = count + 1

	f = 'Output/Q1/' + str(mmmm) + '_part2_20.jpg'
	cv2.imwrite(f, I_out)
	mmmm +=1



