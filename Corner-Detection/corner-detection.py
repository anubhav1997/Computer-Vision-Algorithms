import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from numpy import linalg as LA


def padding(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2, 3)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2, 3)), axis=1)

	# print I.shape 
	I = np.append(np.zeros((filter_size/2, len(I[0]), 3)), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]), 3)), axis=0)
	return I 


def padding2(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2)), axis=1)

	# print I.shape 
	I = np.append(np.zeros((filter_size/2, len(I[0]))), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]))), axis=0)
	return I 


def cal_val(I, x, y, filter1):
	k = 0
	l = 0
	out = 0

	for i in range(x-filter_size//2, x+filter_size//2 + 1):
		for j in range(y-filter_size//2, y+filter_size//2 +1):
			out+=filter1[k,l]*I[i,j]
			l+=1
		k+=1
		l=0

	return out  

def grayscale(I):
    return np.dot(I[...,:3], [0.299, 0.587, 0.114])


def get_filtered_output(I, filter1, filter_size):

	I2 = np.zeros(I.shape)
	I = padding2(I, filter_size)


	i=filter_size/2
	j= filter_size/2

	k = 0
	l = 0

	while(i<(len(I)-filter_size/2)):
		while(j<(len(I[0])-filter_size/2)):
			# for c in range(I.shape[2]):

			I2[k,l] = cal_val(I, i, j, filter1)
			# print 
			j+=1
			l+=1
		i+=1
		j=filter_size/2 
		k+=1
		l=0

	return I2			



def get_M(I, Ix2, Iy2, Ixy, x, y, filter_size):
	k = 0
	l = 0
	out = 0
	M = np.zeros((2,2))
	for i in range(x-filter_size//2, x+filter_size//2 + 1):
		for j in range(y-filter_size//2, y+filter_size//2 +1):
			# out+=filter1[k,l]*I[i,j]
			M[0][0] += Ix2[i][j]
			M[1][0] += Ixy[i][j]
			M[0][1] += Ixy[i][j]
			M[1][1] += Iy2[i][j]
			l+=1
		k+=1
		l=0

	return M

def get_filter(filter_size, sigma, mean =0):
	k = filter_size/2
	pi = 3.14

	filter1 = [[np.exp(-(i*i+j*j)/(2.*sigma*sigma)) for i in xrange(-k,k+1)] for j in xrange(-k,k+1)]
	filter1 = np.array(filter1)/(2.*pi*sigma*sigma)
	filter1 = filter1/np.sum(filter1)
	return filter1



def corner_detection(I, Ix2, Iy2, Ixy, window_size,  gaus_filter_size, threshold):


	# I2 = np.zeros(I.shape)

	sigma = 1.5
	gaus_filter = get_filter(gaus_filter_size, sigma, 0)

	Ix2 = get_filtered_output(Ix2, gaus_filter, gaus_filter_size)
	Iy2 = get_filtered_output(Iy2, gaus_filter, gaus_filter_size)
	Ixy = get_filtered_output(Ixy, gaus_filter, gaus_filter_size)


	filter_size = window_size

	I = padding2(I, filter_size)
	Ix2 = padding2(Ix2, filter_size)
	Iy2 = padding2(Iy2, filter_size)
	Ixy = padding2(Ixy, filter_size)


	i=filter_size/2
	j= filter_size/2

	k = 0
	l = 0
	# K = np.array([0.04, 0.06])
	K = 0.05
	outx = []
	outy = []
	while(i<(len(I)-filter_size/2)):
		while(j<(len(I[0])-filter_size/2)):


			H = get_M(I, Ix2, Iy2, Ixy, i, j, filter_size)
			R = np.linalg.det(H) - K*(np.trace(H)**2)

			if(np.log10(R)>threshold):
				outx.append(k)
				outy.append(l)
			j+=1
			l+=1
		i+=1
		j=filter_size/2 
		k+=1
		l=0
	return I, outx, outy


I = cv2.imread('Corner Detection/yosemite1.jpg')
I.setflags(write=1)

I = grayscale(I)
print I.shape

fx = [[0,0,0], [1, -2, 1], [0, 0, 0]]
fy = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]


fx = np.array(fx)
fy = np.array(fy)

filter_size = 3

Ix = get_filtered_output(I, fx, filter_size)
Iy = get_filtered_output(I, fy, filter_size)


Ix2 = np.multiply(Ix, Ix)
Iy2 = np.multiply(Iy, Iy)
Ixy = np.multiply(Ix, Iy)


window_size = 3
gaus_filter_size = 7


threshold = -2
I, outx, outy = corner_detection(I, Ix2, Iy2, Ixy, window_size, gaus_filter_size, threshold)

plt.imshow(I/255, cmap='gray')
plt.scatter(outy, outx, color='red', lw=0.2, s=4)
plt.show()	

#################### Rotating the image ###############


I = np.rot90(np.rot90(np.rot90(I, axes=(0,1))))

Ix = get_filtered_output(I, fx, filter_size)
Iy = get_filtered_output(I, fy, filter_size)


Ix2 = np.multiply(Ix, Ix)
Iy2 = np.multiply(Iy, Iy)
Ixy = np.multiply(Ix, Iy)


window_size = 3
gaus_filter_size = 7


threshold = -2
I, outx, outy = corner_detection(I, Ix2, Iy2, Ixy, window_size, gaus_filter_size, threshold)

plt.imshow(I/255, cmap='gray')
plt.scatter(outy, outx, color='red', lw=0.2, s=4)
plt.show()	


###################### Resizing the Image #####################


I = cv2.resize(I, (I.shape[1]//2, I.shape[0]//2))

Ix = get_filtered_output(I, fx, filter_size)
Iy = get_filtered_output(I, fy, filter_size)


Ix2 = np.multiply(Ix, Ix)
Iy2 = np.multiply(Iy, Iy)
Ixy = np.multiply(Ix, Iy)


window_size = 3
gaus_filter_size = 7


threshold = -2
I, outx, outy = corner_detection(I, Ix2, Iy2, Ixy, window_size, gaus_filter_size, threshold)

plt.imshow(I/255, cmap='gray')
plt.scatter(outy, outx, color='red', lw=0.2, s=4)
plt.show()	

