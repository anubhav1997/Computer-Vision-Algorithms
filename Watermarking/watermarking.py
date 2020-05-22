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
	for i in range(x-filter_size/2, x+filter_size/2):
		for j in range(y-filter_size/2, y+filter_size/2):
			
			out+=filter1[k,l]*I[i,j, c]
			l+=1
		k+=1
		l=0
	
	return max(out, 0)



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


def thresholding(X):
	threshold = float(np.std(X))*np.sqrt(2*np.log10(X.shape[0]*X.shape[1]*X.shape[2]))

	print threshold

	for i in range(len(X)):
		for j in range(len(X[0])):
			for k in range(len(X[0][0])):
				if(X[i][j][k]>threshold):
					X[i][j][k] = threshold

	return X



I = cv2.imread('image_3.png')
I.setflags(write=1)

I2 = cv2.imread('image_3.png')

out = []
rgbArray = np.zeros(I2.shape, 'uint8')

# for i in range(3):
# 	print 'heyy'
# 	I1 = I[:,:,i]

# 	I22 = I2[:,:,i]
# 	coeffs = pywt.dwt2(I1, 'haar')
# 	cA, (cH, cV, cD) = coeffs


# 	coeffs2 = pywt.dwt2(cA, 'haar')
# 	cA2, (cH2, cV2, cD2) = coeffs2


# 	I22 = cv2.resize(I22, cA2.shape)

# 	u, s, vh = np.linalg.svd(cA2)

# 	u2, s2, vh2 = np.linalg.svd(I22)


# 	alpha = 0.1 
# 	s_new = s + alpha*s2


# 	print u.shape, s_new.shape, s.shape, vh.shape, cA.shape

# 	cA3 = u*s_new*np.transpose(vh)


# 	coeffs = cA3, (cH2, cV2, cD2)
	
# 	cA4 = pywt.idwt2(coeffs, 'haar')
	
# 	coeffs = cA4, (cH, cV, cD)
	
# 	I3 = pywt.idwt2(coeffs, 'haar')
	

# 	print I3.shape 

# 	rgbArray[:,:,i] = I3

print 'heyy'
# I1 = I[:,:,i]

# I22 = I2[:,:,1]

I22 = cv2.imread('image_3.png', 0)
I = I.reshape(I.shape[0], I.shape[1]*3)

coeffs = pywt.dwt2(I, 'haar')
cA, (cH, cV, cD) = coeffs


coeffs2 = pywt.dwt2(cA, 'haar')
cA2, (cH2, cV2, cD2) = coeffs2


I22 = cv2.resize(I22, (cA2.shape[1],cA2.shape[0]) )


u, s, vh = np.linalg.svd(cA2, full_matrices=False)

u2, s2, vh2 = np.linalg.svd(I22, full_matrices=False)

alpha = 0.1 
s_new = s + alpha*s2

cA3 = np.dot(u, np.dot(np.diag(s_new), vh))

coeffs = cA3, (cH2, cV2, cD2)

cA4 = pywt.idwt2(coeffs, 'haar')

coeffs = cA4, (cH, cV, cD)

I3 = pywt.idwt2(coeffs, 'haar')

I3 = I3.reshape((I3.shape[0], I3.shape[1]/3, 3))

cv2.imwrite('Output/Q7/image_3_output.jpg', I3)



I3 = I3.reshape(I3.shape[0], I3.shape[1]*3)

coeffs = pywt.dwt2(I3, 'haar')
cA3, (cH3, cV3, cD3) = coeffs


coeffs2 = pywt.dwt2(cA3, 'haar')
cA23, (cH23, cV23, cD23) = coeffs2

u3, s3, vh3 = np.linalg.svd(cA23, full_matrices=False)

s_new3 = (s3 - s)/alpha

w = np.dot(u3, np.dot(np.diag(s_new3), vh3))


w = w.reshape((w.shape[0], w.shape[1]/3, 3))

cv2.imwrite('Output/Q7/image_3_output_watermark_retrieved.jpg', w)
