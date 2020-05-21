# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import random
from sklearn.svm import SVC
import pandas as pd
import seaborn as sn
from PIL import Image 
# from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
# from sklearn.svm import LinearSVC
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

from skimage.feature import hog
from skimage import data, exposure


def Kmean(X, k, maxiter): 
	C = [X[int(random.random()*len(X))] for i in range(k)]
	# print C
	labels = []

	for i in range(maxiter): 
		labels = []
		for j in range(len(X)):
			temp = []
			for m in range(k):
				temp.append(np.dot(X[j]-C[m], X[j]-C[m])**2)
			
			labels.append(np.argmin(np.array(temp)))
			# print labels
		# C = [X[labels == k1].mean(axis = 0) for k1 in range(k)]
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
				# print mean
				flag = []
				for n in range(len(temp1)):
					flag.append(np.dot(temp1[n]-mean, temp1[n]-mean)**2)
				# print flag 
				C.append(temp1[np.argmin(flag)]) 

		# print C
		# print labels
	return C, labels

def kmean_predict(X, C):

	# labels = []

	# for j in range(len(X)):
	temp = []
	for m in range(len(C)):
		temp.append(np.dot(X-C[m], X-C[m])**2)
	
	# labels.append(np.argmin(np.array(temp)))
	# return labels
	return np.argmin(np.array(temp))


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict



for i in range(5):
	filename = 'cifar-10-batches-py/data_batch_' + str(i+1)
	dict = unpickle(filename)
	if(i==0):
		trainX = dict['data']
		trainY = dict['labels']
	else:

		trainX = np.append(trainX, dict['data'], axis=0)
		trainY = np.append(trainY, dict['labels'], axis=0)


trainX = np.array(trainX)
trainY = np.array(trainY)
print trainX.shape



train = []

for i in range(len(trainX)):

	# # img = Image.fromarray(trainX[i][:1024].reshape(32,32))

	# # print img 
	# img1 = trainX[i][:1024].reshape(32,32,1)
	
	# # f = 'Output/Q4/' + str(3) + '_seeding.jpg'
	# # cv2.imwrite(f, img)


	# img2 = trainX[i][1024:2048].reshape(32,32,1)
	
	# # f = 'Output/Q4/' + str(2) + '_seeding.jpg'
	# # cv2.imwrite(f, img)

	# img3 = trainX[i][2048:3072].reshape(32,32,1)
	
	# # f = 'Output/Q4/' + str(1) + '_seeding.jpg'
	# # cv2.imwrite(f, img)

	# # img = trainX[i].reshape(32,32,3)


	# img = np.append(img3, img2, axis=2)
	# img = np.append(img, img1, axis=2)

	img = np.transpose(np.reshape(trainX[i],(3, 32,32)), (1,2,0))


	# f = 'Output/Q4/' + str(4) + '_seeding.jpg'
	# cv2.imwrite(f, img)


	# x = input()

	# plt.imshow(img)
	# plt.show()	
	# x = trainX[i].reshape(32,32,3)
	# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	
	# x = cv2.resize(x, (x.shape[0]//2, x.shape[1]//2))
	# train.append(x.reshape(x.shape[0]*x.shape[1]))
	train.append(img)


trainX = np.array(train) 
print trainX.shape 

del train 

dict = unpickle('cifar-10-batches-py/test_batch')
testX = dict['data']
testY = dict['labels']

train = []

for i in range(len(testX)):
	x = np.transpose(np.reshape(testX[i],(3, 32,32)), (1,2,0))
	
	# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	
	# x = cv2.resize(x, (x.shape[0]//2, x.shape[1]//2))
	
	# train.append(x.reshape(x.shape[0]*x.shape[1]))
	train.append(x)


testX = np.array(train) 
del train 

n_classes = np.max(trainY)+1


randomize = np.arange(len(trainX))
np.random.shuffle(randomize)
trainX = trainX[randomize]
trainY = trainY[randomize]

# trainX1 = trainX[:len(trainX)//2]
# trainX2 = trainX[len(trainX)//2+1:]

# trainY1 = trainY[:len(trainY)//2]
# trainY2 = trainY[len(trainY)//2+1:]


# del trainX, trainY

################  LBP feature #####################

def hist(I):

	out = np.zeros(256)

	# print out.shape 

	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			x = int(I[i][j])
			# print x 

			out[x]+=1

	return out 





n_points = 8
radius = 1

train = []
y =[]

for k in range(len(trainX)):
	# im = cv2.imread(sample)
	
	# plt.imshow(trainX1[i])
	# plt.show()
	# print trainX1[i].shape 

	# h = hog.compute(trainX1[i])
	
	I = cv2.cvtColor(trainX[k], cv2.COLOR_BGR2GRAY) 
	# for j in range(len())
	h = local_binary_pattern(I, n_points, radius)

	if(k<5):
		# cv2.imsave()
		f = 'Output/Q4/' + str(k) + '_LBP.jpg'
		cv2.imwrite(f, h)
		f = 'Output/Q4/' + str(k) + '_Ori.jpg'
		cv2.imwrite(f, I)

	i = 0
	j=0	
	image_shape = h.shape

	while j<image_shape[1]:
			while i<image_shape[0]:
				I_teemp = h[i:i+8, j:j+8]
				# vals = im.mean(axis=2).flatten()
				# counts, bins = np.histogram(vals, range(257))
				out = hist(I_teemp)

				train.append(out)
				# y.append(trainY[k])
				i = i+8

			j=j+8
			i = 0

	# plt.imshow(h)
	# plt.show()


# C, labels = Kmean(train, 40, 15)


# print C
# print labels
# # x = input()


# # exit(0)

# # for i in range()

# print 'hiii'

# del train

# train = []

# for k in range(len(trainX)):
# 	# im = cv2.imread(sample)
	
# 	# plt.imshow(trainX1[i])
# 	# plt.show()
# 	# print trainX1[i].shape 

# 	# h = hog.compute(trainX1[i])
	
# 	I = cv2.cvtColor(trainX[k], cv2.COLOR_BGR2GRAY) 
# 	# for j in range(len())
# 	h = local_binary_pattern(I, n_points, radius)
# 	i = 0
# 	j=0	
# 	image_shape = h.shape

# 	hist_out = np.zeros(len(C))


# 	while j<image_shape[1]:
# 		while i<image_shape[0]:
# 			I_teemp = h[i:i+8, j:j+8]
# 			# vals = im.mean(axis=2).flatten()
# 			# counts, bins = np.histogram(vals, range(257))
# 			out = hist(I_teemp)

# 			# train.append(out)

# 			labels = kmean_predict(out, C)
# 			# print labels
# 			hist_out[labels]+=1

# 			# y.append(trainY[k])
# 			i = i+8

# 		j=j+8
# 		i = 0
# 	# print hist_out

# 	train.append(hist_out)

# 	# plt.imshow(h)
# 	# plt.show()

# train = np.array(train)
# print train.shape

# # trainY2 = np.array(trainY2)
# # print trainY2.shape 

# print 'here'


# test = []

# for k in range(len(testX)):
# 	# im = cv2.imread(sample)
	
# 	# plt.imshow(trainX1[i])
# 	# plt.show()
# 	# print trainX1[i].shape 

# 	# h = hog.compute(trainX1[i])
	
# 	I = cv2.cvtColor(testX[k], cv2.COLOR_BGR2GRAY) 
# 	# for j in range(len())
# 	h = local_binary_pattern(I, n_points, radius)
# 	i = 0
# 	j=0	
# 	image_shape = h.shape

# 	hist_out = np.zeros(len(C))


# 	while j<image_shape[1]:
# 		while i<image_shape[0]:
# 			I_teemp = h[i:i+8, j:j+8]
# 			# vals = im.mean(axis=2).flatten()
# 			# counts, bins = np.histogram(vals, range(257))
# 			out = hist(I_teemp)

# 			# train.append(out)

# 			labels = kmean_predict(out, C)
# 			hist_out[labels]+=1

# 			# y.append(trainY[k])
# 			i = i+8

# 		j=j+8
# 		i = 0

# 	test.append(hist_out)

# 	# plt.imshow(h)
# 	# plt.show()

# test = np.array(test)
# print test.shape

# print 'hello'


# clf = SVC(max_iter=50000)
# clf.fit(train, trainY)
# preds = clf.predict(test)

# count = 0
# for i in range(len(test)):
# 	if(testY[i]==preds[i]):
# 		count+=1

# print float(count)/len(test)




############### With Hog features ###############

print 'heyyy'

# hog = cv2.HOGDescriptor()

# plt.imshow()

# trainX_hog = []


train = []
y =[]

for k in range(len(trainX)):
	# im = cv2.imread(sample)
	
	# plt.imshow(trainX1[i])
	# plt.show()
	# print trainX1[i].shape 

	# h = hog.compute(trainX1[i])
	
	# I = cv2.cvtColor(trainX1[k], cv2.COLOR_BGR2GRAY) 
	# for j in range(len())
	# h = local_binary_pattern(I, n_points, radius)
	fd, h = hog(trainX[k], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
	
	if(k<5):
		# cv2.imsave()
		f = 'Output/Q4/' + str(k) + '_HOG1.jpg'
		cv2.imwrite(f, fd)

	i = 0
	j=0	
	image_shape = h.shape

	while j<image_shape[1]:
			while i<image_shape[0]:
				I_teemp = h[i:i+8, j:j+8]
				# vals = im.mean(axis=2).flatten()
				# counts, bins = np.histogram(vals, range(257))
				out = hist(I_teemp)
				# print out 

				train.append(out)
				# y.append(trainY[k])
				i = i+8

			j=j+8
			i = 0

	# plt.imshow(h)
	# plt.show()


C, labels = Kmean(train, 40, 20)


print C 
# exit(0)
# print labels

# for i in range()

print 'hiii'

del train

train = []

for k in range(len(trainX)):
	# im = cv2.imread(sample)
	
	# plt.imshow(trainX1[i])
	# plt.show()
	# print trainX1[i].shape 

	# h = hog.compute(trainX1[i])
	
	# I = cv2.cvtColor(trainX2[k], cv2.COLOR_BGR2GRAY) 
	# for j in range(len())
	# h = local_binary_pattern(I, n_points, radius)
	fg, h = hog(trainX[k], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

	i = 0
	j = 0	
	image_shape = h.shape

	hist_out = np.zeros(len(C))

	while j<image_shape[1]:
		while i<image_shape[0]:
			I_teemp = h[i:i+8, j:j+8]
			# vals = im.mean(axis=2).flatten()
			# counts, bins = np.histogram(vals, range(257))
			out = hist(I_teemp)

			# train.append(out)

			labels = kmean_predict(out, C)
			hist_out[labels]+=1

			# y.append(trainY[k])
			i = i+8

		j=j+8
		i = 0

	train.append(hist_out)

	# plt.imshow(h)
	# plt.show()

train = np.array(train)
print train.shape

trainY = np.array(trainY)
print trainY.shape 

print 'here'


test = []

for k in range(len(testX)):
	# im = cv2.imread(sample)
	
	# plt.imshow(trainX1[i])
	# plt.show()
	# print trainX1[i].shape 

	# h = hog.compute(trainX1[i])
	
	# I = cv2.cvtColor(testX[k], cv2.COLOR_BGR2GRAY) 
	# for j in range(len())
	fd, h = hog(testX[k], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

	# h = local_binary_pattern(I, n_points, radius)
	i = 0
	j=0	
	image_shape = h.shape

	hist_out = np.zeros(len(C))


	while j<image_shape[1]:
		while i<image_shape[0]:
			I_teemp = h[i:i+8, j:j+8]
			# vals = im.mean(axis=2).flatten()
			# counts, bins = np.histogram(vals, range(257))
			out = hist(I_teemp)

			# train.append(out)

			labels = kmean_predict(out, C)
			hist_out[labels]+=1

			# y.append(trainY[k])
			i = i+8

		j=j+8
		i = 0

	test.append(hist_out)

	# plt.imshow(h)
	# plt.show()

test = np.array(test)
print test.shape

print 'hello'


clf = SVC(max_iter=50000)
clf.fit(train, trainY)
preds = clf.predict(test)

count = 0
for i in range(len(test)):
	if(testY[i]==preds[i]):
		count+=1

print float(count)/len(test)

