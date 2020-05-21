import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np



def padding(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2, 3)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2, 3)), axis=1)

	I = np.append(np.zeros((filter_size/2, len(I[0]), 3)), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]), 3)), axis=0)
	return I 


def cal_val(I, x, y, c, filter1):
	k = 0
	l = 0
	out = 0
	# print x,y,c

	for i in range(x-filter_size//2, x+filter_size//2 + 1):
		for j in range(y-filter_size//2, y+filter_size//2 +1):
			# print i,j, k, l

			out+=filter1[k,l]*I[i,j, c]
			l+=1
		k+=1
		l=0

	# print out, I[x,y,c]

	# x = input()

	return out  


def get_filter(filter_size, sigma, mean =0):
	k = filter_size/2
	pi = 3.14

	filter1 = [[np.exp(-(i*i+j*j)/(2.*sigma*sigma)) for i in xrange(-k,k+1)] for j in xrange(-k,k+1)]
	filter1 = np.array(filter1)/(2.*pi*sigma*sigma)
	filter1 = filter1/np.sum(filter1)
	return filter1



I = cv2.imread('image_3.png')
I.setflags(write=1)

filter_size = input("Enter the filter size: ")
sigma = input("Enter sigma value: ")


filter1 = get_filter(filter_size, sigma, 0)
print filter1.shape
print filter1

I2 = np.zeros(I.shape)

I = padding(I, filter_size)
# print I.shape 


i = filter_size/2
j = filter_size/2

k = 0
l = 0

while(i<(len(I)-filter_size/2)):
	while(j<(len(I[0])-filter_size/2)):
		for c in range(I.shape[2]):
			I2[k,l, c] = cal_val(I, i, j,c, filter1)
		j+=1
		l+=1
	i+=1
	j=filter_size/2 
	k+=1
	l=0

cv2.imwrite(('Output/Q3/image_3_gaussian_%d_%d.jpg'% (filter_size, sigma)), I2)


if(filter_size==15):
	I = cv2.imread('image_3.png')
	
	I3 = cv2.GaussianBlur(I,(filter_size,filter_size),sigma, sigma, cv2.BORDER_CONSTANT)
	cv2.imwrite(('Output/Q5/image_1_gaussian_%d_cv2.jpg'% filter_size), I3)
	cv2.imwrite(('Output/Q5/image_1_gaussian_%d_diff.png'% filter_size), (I3-I2))
	

