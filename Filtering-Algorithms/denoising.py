import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import pywt
import random 

def padding(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2, I.shape[2])),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2, I.shape[2])), axis=1)

	# print I.shape 
	I = np.append(np.zeros((filter_size/2, len(I[0]), I.shape[2])), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]), I.shape[2])), axis=0)
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

	return max(out,0)  



def noise(I, noise_level):
	
	total_pixels = I.shape[0]*I.shape[1]
	noise_pixels = noise_level/100*total_pixels

	count = 0 
	black = 0


	x_cord = random.sample(range(1, I.shape[0]), noise_level*I.shape[0]/100)
	y_cord = random.sample(range(1, I.shape[1]), noise_level*I.shape[1]/100)

	I[x_cord, y_cord, :] = 0


	x_cord = random.sample(range(1, I.shape[0]), noise_level*I.shape[0]/100)
	y_cord = random.sample(range(1, I.shape[1]), noise_level*I.shape[1]/100)

	I[x_cord, y_cord, :] = 255

	return I 

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


def thresholding(X):
	threshold = float(np.std(X))*np.sqrt(2*np.log(X.shape[0]*X.shape[1]*X.shape[2]))

	# print threshold

	# threshold = 127

	threshold = 0

	for i in range(len(X)):
		for j in range(len(X[0])):
			for k in range(len(X[0][0])):
				if(X[i][j][k]>threshold):
					X[i][j][k] = 0

	return X



def med(A):
	A.sort()
	if(len(A)%2==0):
		return (A[len(A)//2]+A[len(A)//2 + 1])/2
	else:
		return A[(len(A)+1)//2]


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


def filtering(I, filter_size):
	
	I2 = np.zeros(I.shape)
	I = padding(I, filter_size)

	i=filter_size/2
	j= filter_size/2

	k = 0
	l = 0

	while(i<(len(I)-filter_size/2)):
		while(j<(len(I[0])-filter_size/2)):
			for c in range(I.shape[2]):

				I2[k,l, c] = median(I, i, j,c)
			# print 
			j+=1
			l+=1
		i+=1
		j=filter_size/2 
		k+=1
		l=0
	return I2

I = cv2.imread('image_3.png')
I.setflags(write=1)

filter_size = 3
filter1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]


# print filter1

filter1 = np.array(filter1)

I2 = np.zeros(I.shape)
I = padding(I, filter_size)

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


I = cv2.imread('image_3.png')


I3 = noise2(I, 10)


cv2.imwrite('Output/Q6/image_3_noisy.jpg', I3)
cv2.imwrite('Output/Q6/image_3_lap.jpg', I2)


I = I3-I2

cv2.imwrite('Output/Q6/image_3_I_dash.jpg', I)

coeff = pywt.dwt2(I, 'haar')
cA, (cH, cV, cD) = coeff

coeff2 = pywt.dwt2(cA, 'haar')
cA2, (cH2, cV2, cD2) = coeff2

# coeff3 = pywt.dwt2(cA2, 'haar')
# cA3, (cH3, cV3, cD3) = coeff3


# cD3 = thresholding(cD3)
# cV3 = thresholding(cV3)
# cH3 = thresholding(cH3)

cD = thresholding(cD)
cV = thresholding(cV)
cH = thresholding(cH)

cD2 = thresholding(cD2)
cV2 = thresholding(cV2)
cH2 = thresholding(cH2)

# filter2 = []

cA = filtering(cA, 7)
cA2 = filtering(cA2, 7)

# cA = cv2.medianBlur(cA.astype('uint8'), filter_size)

# cA2 = cv2.medianBlur(cA2.astype('uint8'), filter_size)

# coeff3 = cA3, (cH3, cV3, cD3)
# cA2 = pywt.idwt2(coeff3, 'haar')

print cA2.shape, cH2.shape, cV2.shape, cD2.shape 

# cA2 = cA2[:,:,0]

coeff2 = cA2.reshape(cA2.shape[0], cA2.shape[1], 1), (cH2, cV2, cD2)

cA = pywt.idwt2(coeff2, 'haar')
coeffs = cA, (cH, cV, cD)

I = pywt.idwt2(coeffs, 'haar')

cv2.imwrite('Output/Q6/image_3_output.jpg', I)
