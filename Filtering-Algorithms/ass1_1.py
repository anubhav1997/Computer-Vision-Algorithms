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

	# print I.shape 
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


I = cv2.imread('image_1.jpg')
I.setflags(write=1)

filter_size = input("Enter the filter size: ")

filter1 = 1/float(filter_size*filter_size)*np.ones((filter_size, filter_size))

I2 = np.zeros(I.shape)

I = padding(I, filter_size)
# print I.shape 


i=filter_size/2
j= filter_size/2

k = 0
l = 0

while(i<(len(I)-filter_size/2)):
	while(j<(len(I[0])-filter_size/2)):
		for c in range(I.shape[2]):

			I2[k,l, c] = cal_val(I, i, j,c, filter1)
		# print 
		j+=1
		l+=1
	i+=1
	j=filter_size/2 
	k+=1
	l=0

cv2.imwrite(('Output/Q1/image_1_mean%d.jpg'% filter_size), I2)


if(filter_size==15):
	I = cv2.imread('image_1.jpg')
	
	filt = np.ones((filter_size,filter_size),np.float32)/(filter_size*filter_size)
	I3 = cv2.filter2D(I,-1, filt)

	cv2.imwrite(('Output/Q5/image_1_mean_%d_cv2.jpg'% filter_size), I3)
	cv2.imwrite(('Output/Q5/image_1_mean_%d_diff.png'% filter_size), abs(I2-I3))
	

