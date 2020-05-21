
########## SEEDING ALGORITHM ############################


def seeded(I, I_out, x, y, avg):


	print np.absolute(I[x][y][2]-avg[2])
	print np.absolute(I[x][y][0]-avg[0])
	print np.absolute(I[x][y][1]-avg[1])


	if(x<I.shape[0] and y<I.shape[1] and x>0 and y>0 and np.absolute(I[x][y][0]-avg[0])<123 and np.absolute(I[x][y][1]-avg[1])<100 and np.absolute(I[x][y][2]-avg[2])<100  and I_out[x][y]!=1):
		I_out[x][y] = 1
		I_out = seeded(I,I_out, x+1, y, avg)
		I_out = seeded(I,I_out, x+1, y+1, avg)
		I_out = seeded(I,I_out, x-1, y-1, avg)
		I_out = seeded(I,I_out, x, y+1, avg)
		I_out = seeded(I,I_out, x-1, y+1, avg)
		I_out = seeded(I,I_out, x+1, y-1, avg)
		I_out = seeded(I,I_out, x-1, y, avg)
		I_out = seeded(I,I_out, x, y-1, avg)


	return I_out


def comp(X, Y, e):
	
	if(abs(X[0]-Y[0])<e and abs(X[1]-Y[1])<e and abs(X[2]-Y[2])<e):
		return True 
	else:
		return False 

def seeding2(I, I_out, x, y, e):
	arr = []
	arr.append((x,y))
	# l = len(arr)
	print I.shape 
	
	val = I[x][y]



	while(len(arr)>0):

		x1, y1 = arr.pop()

		if(comp(I[x1][y1], val, e)):
			I_out[x1][y1] = 1
	
			for i in range(-1,2):
				for j in range(-1,2):
					if((x1+i)<I.shape[0] and (x1+i)>0 and (y1+j)<I.shape[1] and (y1+j)>0):

						if(I_out[x1+i][y1+j]==0 and (i,j)!=(1,1)):
							arr.append((x1+i,y1+j))

	return I_out


I = cv2.imread('Q3-faces/face3.jpg')


# I = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
I = I.astype('float64')

I_out = np.zeros((I.shape[0], I.shape[1]))
# I_out = seeded(I, I_out, 247, 424, (229,184,143))
I_out = seeding2(I, I_out, 247, 424, 100)



I[:,:,0] = I_out*I[:,:,0]
I[:,:,1] = I_out*I[:,:,1]
I[:,:,2] = I_out*I[:,:,2]

f = 'Output/Q3/' + str(3) + '_seeding.jpg'
cv2.imwrite(f, I)
# plt.imshow(I)
# plt.show()



I = cv2.imread('Q3-faces/face1.jpg')
# print I
# plt.imshow(I)
# plt.show()

# I = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
I = I.astype('float64')

I_out = np.zeros((I.shape[0], I.shape[1]))
# I_out = seeded(I, I_out, 247, 424, (229,184,143))
I_out = seeding2(I, I_out, 456, 570, 100)

# for i in range(max_iter):

# print I_out

I[:,:,0] = I_out*I[:,:,0]
I[:,:,1] = I_out*I[:,:,1]
I[:,:,2] = I_out*I[:,:,2]

f = 'Output/Q3/' + str(1) + '_seeding.jpg'
cv2.imwrite(f, I)
# plt.imshow(I)
# plt.show()




I = cv2.imread('Q3-faces/face2.jpg')
# print I
plt.imshow(I)
plt.show()

# I = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
I = I.astype('float64')

I_out = np.zeros((I.shape[0], I.shape[1]))
# I_out = seeded(I, I_out, 247, 424, (229,184,143))
I_out = seeding2(I, I_out, 526, 231,51)


I[:,:,0] = I_out*I[:,:,0]
I[:,:,1] = I_out*I[:,:,1]
I[:,:,2] = I_out*I[:,:,2]

f = 'Output/Q3/' + str(2) + '_seeding.jpg'
cv2.imwrite(f, I)
# plt.imshow(I)
# plt.show()









I = cv2.imread('Q3-faces/face4.jpg')


# I = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
# I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
I = I.astype('float64')

I_out = np.zeros((I.shape[0], I.shape[1]))
# I_out = seeded(I, I_out, 247, 424, (229,184,143))
I_out = seeding2(I, I_out, 251,106, 50)



I[:,:,0] = I_out*I[:,:,0]
I[:,:,1] = I_out*I[:,:,1]
I[:,:,2] = I_out*I[:,:,2]

f = 'Output/Q3/' + str(0) + '_seeding.jpg'
cv2.imwrite(f, I)

