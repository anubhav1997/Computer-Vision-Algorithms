import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np



def bilinear_interpolation(I, size_new):
	
	I2 = np.zeros(size_new)
	
	x = I.shape[0]/float(size_new[0])
	y = I.shape[1]/float(size_new[1])


	for i in range(size_new[0]):
		for j in range(size_new[1]):
			x_new = i*x
			y_new = j*y
			
			x_next = min(int(i*x) +1, I.shape[0]-1)
			y_next = min(int(j*y) +1, I.shape[1]-1)
			
			x_prev = max(int(i*x), 0)
			y_prev = max(int(j*y), 0)

			a = int(i*x) - x_new
			b = int(j*y) - y_new

			for k in range(I.shape[2]):
				I2[i, j, k] = (1-a)*(1-b)*I[x_prev, y_prev, k] + (1-a)*b*I[x_next,y_prev, k] + a*(1-b)*I[x_prev,y_next, k] + a*b*I[x_next, y_next, k]

	return I2


def resize(I, size_new):
	

	I2 = -1*np.zeros(size_new)
	
	x = float(size_new[0])/I.shape[0]
	y = float(size_new[1])/I.shape[1]


	for i in range(len(I)):
		for j in range(len(I[0])):
			for k in range(len(I[0][0])):

				I2[int(x*i)][int(j*y)][k] = I[i][j][k]



	I3 = I2.copy()
	for c in range(len(I2[0][0])):

		for i in range(len(I2)):
			temp = (0,0,0)
			for j in range(len(I2[0])):
				if(I3[i][j][c]==-1 and temp!=(0,0,0)):
					k = j+1
					while(k<len(I2[0]) and I2[i][k][c]==-1):
						k+=1

					if(k<len(I2[0])):

						I3[i][j][c] = (I2[i][temp[2]][c]*(k-j) +  I2[i][k][c]*(j-temp[2]))/(k-temp[2])  

				else:
					temp = (I2[i][j][c],i,j)	


	for c in range(len(I2[0][0])):

		for j in range(len(I2[0])):
			temp = (0,0,0)
			for i in range(len(I2)):
				if(I3[i][j][c]==-1 and temp!=(0,0,0)):
					k = i+1
					while(k<len(I2) and I2[k][j][c]==-1):
						k+=1

					if(k<len(I2)):

						I3[i][j][c] = (I2[temp[temp[1]]][j][c]*(k-i) +  I2[k][j][c]*(i-temp[1]))/(k-temp[1])  

				else:
					temp = (I2[i][j],i,j)	



	for c in range(len(I2[0][0])):
		for j in range(len(I2[0])):
			for i in range(len(I2)):
				if(I3[i][j][c]==-1 and i>=1 and j>=1 and i<len(I2)-1 and j<len(I2[0]-1)):
					I3[i][j][c] = (I2[i+1][j+1][c] + I2[i-1][j+1][c] + I2[i+1][j-1][c] + I2[i-1][j-1][c])/4
	

	for c in range(len(I2[0][0])):
		for j in range(len(I2[0])):
			for i in range(len(I2)):
				if(I3[i][j][c]==-1):
					print 'hiii'

	return I3


def resize2(I, size_new):
	

	I2 = -1*np.zeros(size_new)
	
	x = float(size_new[0])/I.shape[0]
	y = float(size_new[1])/I.shape[1]


	for i in range(len(I)):
		for j in range(len(I[0])):
			for k in range(len(I[0][0])):
				I2[int(x*i)][int(j*y)][k] = I[i][j][k]


	for c in range(len(I2[0][0])):
		for i in range(len(I2)):
			temp = (0,0,0)
			for j in range(len(I2[0])):
				if(I2[i][j][c]==-1 and temp!=(0,0,0)):
					k = j+1
					while(k<len(I2[0]) and I2[i][k][c]==-1):
						k+=1
					if(k<len(I2[0])):
						I2[i][j][c] = (I2[i][temp[2]][c]*(k-j) +  I2[i][k][c]*(j-temp[2]))/(k-temp[2])  
				else:
					temp = (I2[i][j][c],i,j)	

	for c in range(len(I2[0][0])):
		for j in range(len(I2[0])):
			temp = (0,0,0)
			for i in range(len(I2)):
				if(I2[i][j][c]==-1 and temp!=(0,0,0)):
					k = i+1
					while(k<len(I2) and I2[k][j][c]==-1):
						k+=1
					if(k<len(I2)):
						I2[i][j][c] = (I2[temp[temp[1]]][j][c]*(k-i) +  I2[k][j][c]*(i-temp[1]))/(k-temp[1])  
				else:
					temp = (I2[i][j],i,j)	


	for c in range(len(I2[0][0])):
		for j in range(len(I2[0])):
			for i in range(len(I2)):
				if(I2[i][j][c]==-1 and i>=1 and j>=1 and i<len(I2)-1 and j<len(I2[0]-1)):
					I2[i][j][c] = (I2[i+1][j+1][c] + I2[i-1][j+1][c] + I2[i+1][j-1][c] + I2[i-1][j-1][c])/4
	


	for c in range(len(I2[0][0])):
		for j in range(len(I2[0])):
			for i in range(len(I2)):
				if(I2[i][j][c]==-1):
					print 'hiii'

	return I2

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


def gaussian(filter_size, sigma, mean =0):
	k = filter_size/2
	pi = 3.14

	filter1 = [[np.exp(-(i*i+j*j)/(2.*sigma*sigma)) for i in xrange(-k,k+1)] for j in xrange(-k,k+1)]
	filter1 = np.array(filter1)/(2.*pi*sigma*sigma)
	filter1 = filter1/np.sum(filter1)
	
	return filter1

def gaussian2():
	filt = [[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]]
	# return 1/16.*np.array(filt)
	return np.array(filt)/float(np.sum(filt))

def laplacian():
	return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])



I = cv2.imread('image_3.png')
I.setflags(write=1)

filter_size = 5

filter1 = gaussian2()
# filter2 = laplacian()


levels = input('Levels: ')


for m in range(levels):

	I2 = np.zeros(I.shape)
	I = padding(I, filter_size)

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


	cv2.imwrite(('Output/Q4/image_3_gaussian_pyr_%d.jpg'% m), I2)
	
	# I = cv2.resize(I2, (0,0), fx=0.5, fy=0.5) 
	
	# I3 = cv2.resize(I, (0,0), fx=2, fy=2) 

	I = resize2(I2, (len(I2)//2, len(I2[0])//2, 3))

	I3 = bilinear_interpolation(I, (len(I)*2, len(I[0])*2, 3))


	I4 = I3 - I2 
	cv2.imwrite(('Output/Q4/image_3_laplacian_pyr_%d.jpg'% m), I4)

