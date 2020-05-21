

##################### Thresholding ###########################

def thresholding(I, I_out, lower, upper, e):
	# x = np.argwhere(I>thresh_l and I<thresh_h)

	# I_out[x] = 1
	# I_out[np.argwhere(I<thresh_h)] = 1
	
	for i in range(I.shape[0]):
		for j in range(I.shape[1]):
			if(I[i][j][0]>lower[0] and I[i][j][1]>lower[1] and I[i][j][2]>lower[2] and I[i][j][0]<upper[0] and I[i][j][1]<upper[1] and I[i][j][2]<upper[2]):
				I_out[i][j]=1

	return I_out

# thresh_h = [255,223,196]
# thresh_l = [165,114,87]




lower = np.array([0, 10, 60], dtype = "uint8") 
upper = np.array([20, 150, 255], dtype = "uint8")


# max_iter = 10


mmmm = 0

for filename in glob.glob('Q3-faces/*'): 

	I = cv2.imread(filename)
	# print I
	I = I.astype('float64')
	
	# I = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
	I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
	
	I_out = np.zeros((I.shape[0], I.shape[1]))
	# I_out = seeded(I, I_out, 247, 424, (229,184,143))
	I_out = thresholding(I, I_out, lower, upper, 10)
	
	# for i in range(max_iter):

	print I_out

	f = 'Output/Q3/' + str(mmmm) + '_thresholding2.jpg'
	cv2.imwrite(f, I_out*255)

	I[:,:,0] = I_out*I[:,:,0]
	I[:,:,1] = I_out*I[:,:,1]
	I[:,:,2] = I_out*I[:,:,2]

	f = 'Output/Q3/' + str(mmmm) + '_thresholding.jpg'
	cv2.imwrite(f, I)
	mmmm+=1
	# plt.imshow(I)
	# plt.show()
