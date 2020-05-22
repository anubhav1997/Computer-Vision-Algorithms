import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np


def padding2(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2)), axis=1)

	# print I.shape 
	I = np.append(np.zeros((filter_size/2, len(I[0]))), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]))), axis=0)
	return I 


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

def padding(I, filter_size):
	
	I = np.append(np.zeros((len(I), filter_size/2, 3)),I, axis=1)
	I = np.append(I, np.zeros((len(I), filter_size/2, 3)), axis=1)

	# print I.shape 
	I = np.append(np.zeros((filter_size/2, len(I[0]), 3)), I, axis=0)
	I = np.append(I, np.zeros((filter_size/2, len(I[0]), 3)), axis=0)
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
def grayscale(I):
    return np.dot(I[...,:3], [0.299, 0.587, 0.114])

def circular_hough_transform(I):
	s = I.shape 
	max_r = np.sqrt(s[0]**2 + s[1]**2)
	max_r = min(s[0], s[1])//2
	
	cos = [np.cos(np.deg2rad(i)) for i in range(1,361)]
	sin = [np.sin(np.deg2rad(i)) for i in range(1,361)]

	cos = np.array(cos)
	sin = np.array(sin)

	edges = np.argwhere(I[:,:])
	out = []

	for i in range(2,int(max_r)):
		print i 

		
		threshold = 1000/i + 5*i


		acc_array = np.zeros((s[0], s[1]))

		for (j,k) in edges:
		
			# theta = 0

			# x1 = j*np.ones(cos.shape) - i*cos
			# x2 = k*np.ones(sin.shape) - i*sin
			
			# # y1 = x1#+int(max_r)*np.ones(cos.shape)
			# # y2 = x2#+int(max_r)*np.ones(cos.shape)

			# x1 = x1.astype('int')
			# x2 = x2.astype('int')

			# if(x1>0 and x1<s[0] and x2>0 and x2<s[1]):

			# 	acc_array[x1, x2]+=1


			for l in range(0,360):
				x1 = j - i*np.cos(np.deg2rad(l))
				x2 = k - i*np.sin(np.deg2rad(l))
				# acc_array[int(x1)+int(max_r)][int(x2)+int(max_r)][i]+=1

				
				x1 = x1.astype('int')
				x2 = x2.astype('int')
				
				if(x1>0 and x1<s[0] and x2>0 and x2<s[1]):

					acc_array[x1,x2]+=1


		indexes = np.argwhere(acc_array>threshold)
		# out.append(indexes)

		# print indexes
		print 'threshold', threshold
		print 'max', np.max(acc_array)
		print 'detected', len(indexes)

		for l in range(len(indexes)):

			x1 = indexes[l,0] + i*cos
			x2 = indexes[l,1] + i*sin

			plt.scatter(x2, x1, color='red', lw=0.2, s=4)

	return out


def thresholding_acc(acc_array, threshold):

	print acc_array.shape
	s = acc_array.shape
	i = 0
	j =0
	k = 0

	for i in range(s[0]):
		for j in range(s[1]):
			for k in range(s[2]):
				if(acc_array[i][j][k]>threshold):
					print i,j,k
	# return 

I = cv2.imread('Q1.jpeg')

I.setflags(write=1)

I = grayscale(I)
I = I.astype('uint8')
filter_size = 3

I = cv2.resize(I, (I.shape[1]//4,I.shape[0]//4))

gaus_filter = gaussian(filter_size, 1.5, 0)
I = get_filtered_output(I, gaus_filter, filter_size)
I = I.astype('uint8')
cv2.imwrite('filtered.jpg', I)


I = cv2.Canny(I,75,110)
cv2.imwrite('edge.jpg', I)


plt.imshow(I, cmap='gray')

indexes = circular_hough_transform(I)
plt.show()


# # threshold = 1065
# # print np.max(acc_array)
# # print np.min(acc_array)

# # indexes = np.argwhere(acc_array>threshold)

# print 'max_radius', np.max(indexes[:,2])

# print indexes

# # thresholding_acc(acc_array, threshold)
# print indexes.shape 

# cos = [np.cos(np.deg2rad(i)) for i in range(1,361)]
# sin = [np.sin(np.deg2rad(i)) for i in range(1,361)]

# cos = np.array(cos)
# sin = np.array(sin)


# plt.imshow(I, cmap='gray')

# for i in range(len(indexes)):

# 	x1 = indexes[i,0] + indexes[i,2]*cos
# 	x2 = indexes[i,1] + indexes[i,2]*sin

# 	plt.scatter(x2, x1, color='red', lw=0.2, s=4)


# plt.show()

# # plt.imshow(I, cmap='gray')
# # plt.Circle((indexes[1,1], indexes[1,0]), indexes[1,2], color='yellow')
# # plt.show()	

# # circle1 = plt.Circle((indexes[:,1], indexes[:,0]), indexes[:,2], color='red')


# # fig, ax = plt.subplots()
# # ax.add_artist(circle1)
# # plt.imshow(I, cmap='gray')
# # plt.show()
# # fig.savefig('plotcircles.png')

