import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import random
	

def padding(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2, 3)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2, 3)), axis=1)

	I = np.append(np.zeros((filter_size/2, len(I[0]), 3)), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]), 3)), axis=0)
	return I 



def med(A):
	A.sort()
	if(len(A)%2==0):
		return (A[len(A)//2 -1 ]+A[len(A)//2 ])/2
	else:
		return A[(len(A)+1)//2 -1]

def median(I, x, y, channel):
	k = 0
	l = 0
	out = []
	for i in range(x-filter_size//2, x+filter_size//2+1):
		for j in range(y-filter_size//2, y+filter_size//2+1):

			out.append(I[i,j, channel])
			l+=1
		k+=1
		l=0
	
	return med(out)

def noise2(I, noise_level):
	x = I.shape[0]
	y = I.shape[1]


	total_pixels = I.shape[0]*I.shape[1]
	noise_pixels = noise_level*total_pixels/100

	I = I.reshape((total_pixels, 3))
	coords = random.sample(range(0, total_pixels), noise_level*I.shape[0]/100)
	I[coords[0:len(coords)//2], :] = 0
	I[coords[len(coords)//2:], :] = 255
	 
	I = I.reshape((x,y,3))
	return I 



I = cv2.imread('image_2.png')
I.setflags(write=1)

filter_size = input("Enter the filter size: ")
noise_level = input("Enter the percentage of noise to be added: ")

I = noise2(I, noise_level)
I3333 = I.copy()

I2 = I.copy()

cv2.imwrite(('Output/Q2/image_2_noise_%d.png'% noise_level), I)
I = padding(I, filter_size)

i = filter_size/2 
j = filter_size/2

k = 0
l = 0

while(i<(len(I)-filter_size/2 )):
	while(j<(len(I[0])-filter_size/2 )):
		for c in range(I.shape[2]):
			I2[k,l, c] = median(I, i, j, c)
		
		j+=1
		l+=1
	i+=1
	j=filter_size/2
	k+=1
	l=0


cv2.imwrite(('Output/Q2/image_2_median_%d_%d.png'% (filter_size, noise_level)), I2)


###### comparing with open CV ####################

if(filter_size==11):
	# I = cv2.imread('image_2.png')
	# I.setflags(write=1)
	# I = noise2(I, noise_level)

	final = cv2.medianBlur(I3333, filter_size)
	cv2.imwrite(('Output/Q5/image_2_median_%d_cv2.png'% filter_size), final)
	cv2.imwrite(('Output/Q5/image_2_median_%d_diff.png'% filter_size), I2-final)
