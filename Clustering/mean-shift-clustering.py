from sklearn.cluster import MeanShift
import numpy as np
import sys
import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import random 
from sklearn import metrics
import glob 
import cv2 
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs


mmmm = 0

for filename in glob.glob('Q2-images/*'): #assuming gif
	X1 = []

	I = cv2.imread(filename)
	I = cv2.resize(I, (0,0), fx=0.5, fy=0.5) 

	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			X1.append(I[i][j][:].tolist())

	bandwidth = estimate_bandwidth(X1, quantile=0.25, n_samples=10)
	clustering = MeanShift(bandwidth=2, bin_seeding=True).fit(X1)

	labels = clustering.labels_
	C = clustering.cluster_centers_
	X1 = np.array(X1)

	labels = np.array(labels)

	count = 0
	I_out = np.zeros(I.shape)
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			
			I_out[i][j][:] = C[labels[count]]
			count = count + 1

	f = 'Output/Q2/' + str(mmmm) + '_2.jpg'
	cv2.imwrite(f, I_out)
	mmmm +=1


