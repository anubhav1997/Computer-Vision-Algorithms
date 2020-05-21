import numpy as np
import cv2
from matplotlib import pyplot as plt
import math



 ################### RAnsac#############################################
def Ransac(src_pts,dst_pts,iter=2500,treshold=50):
	error = np.zeros(iter) 
	M1 = []
	x = np.concatenate((src_pts,dst_pts),axis = 1)	
	for i in range(iter):
		idx = np.random.randint(len(src_pts), size=4)
		x_new = x[idx,:]
		src_pts_new = x_new[:,0]
		dst_pts_new = x_new[:,1]
		M1.append(my_Dlt_algo(src_pts_new,dst_pts_new))
		idx = np.random.randint(len(src_pts),size=len(src_pts)-4)
		pts= x[idx,:]
		pts_src = np.float32(pts[:,0]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts_src,M1[i])
		for k in range(len(pts)):
			error[i] = error[i] + math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2)			
	#print error,error[error.argmin()]	
	final_matrix = M1[error.argmin()]
	pts= x
	mask = []	
	pts_src = np.float32(pts[:,0]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts_src,final_matrix)
	for k in range(len(pts)):
		#print math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2) 	
		if(math.sqrt((pts[k][1][0]-dst[k][0][0])**2 + (pts[k][1][1]-dst[k][0][1])**2) < treshold):
			mask.append(1)
		else:
			mask.append(0)		
	return final_matrix,mask	

############################# DLT ALGO ##################################################3

def my_Dlt_algo(src_pts,dst_pts):
	a = np.zeros((2*len(src_pts),9))
	for i in range(len(src_pts)):
		a[2*i] = [-src_pts[i][0],-src_pts[i][1],-1,0,0,0,dst_pts[i][0]*src_pts[i][0],dst_pts[i][0]*src_pts[i][1],dst_pts[i][0]]
		a[2*i+1] = [0,0,0,-src_pts[i][0],-src_pts[i][1],-1,dst_pts[i][1]*src_pts[i][0],dst_pts[i][1]*src_pts[i][1],dst_pts[i][1]]
	u,s,v = np.linalg.svd(a)
	h = v[8].reshape((3,3))
	s = np.linalg.svd(a)
	return h/h[2,2]	

img1 = cv2.imread('Q3/test1.jpeg',0)          # queryImage
img2 = cv2.imread('Q3/collage.jpg',0) # trainImage

print img1.shape 


########################## Shift script taken from opencv #########################################################3

sift = cv2.xfeatures2d.SIFT_create()

######################### finding  the keypoints and descriptors with SIFT ###################################

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
dist = 0.9
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
	dist = dist+0.0001


src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

imgc_1 = cv2.imread('Q3/test1.jpeg')          # queryImage
imgc_2 = cv2.imread('Q3/collage.jpg') # trainImage

for i in range(len(src_pts)):
	cv2.line(imgc_1,(src_pts[i][0][0],src_pts[i][0][1]),(src_pts[i][0][0],src_pts[i][0][1]),(0,0,255),5)
	cv2.line(imgc_2,(dst_pts[i][0][0],dst_pts[i][0][1]),(dst_pts[i][0][0],dst_pts[i][0][1]),(0,0,255),5)
#img4 = cv2.drawMatches(img1,src_pts)
imgc_1 = imgc_1[:,:,::-1]
plt.imshow(imgc_1),plt.show()
imgc_2 = imgc_2[:,:,::-1]
plt.imshow(imgc_2),plt.show()


h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) # taking corner points

M1,matchesMask = Ransac(src_pts,dst_pts,2000,10) ########### appling Ransac and DLT algo
print M1

dst= cv2.perspectiveTransform(pts,M1)  
      
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

####################3 plotting final images ######################################
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
p1 = (dst[0][0][0],dst[0][0][1])
p2 = (dst[1][0][0],dst[1][0][1])
p3 = (dst[2][0][0],dst[2][0][1])
p4 = (dst[3][0][0],dst[3][0][1])
img2 = cv2.imread('Q3/collage.jpg')
cv2.line(img2,p1,p2,(0,0,255),15)
cv2.line(img2,p2,p3,(0,0,255),15)
cv2.line(img2,p3,p4,(0,0,255),15)
cv2.line(img2,p4,p1,(0,0,255),15)
img2 = img2[:,:,::-1]
plt.imshow(img3,),plt.show()
plt.imshow(img2),plt.show()

