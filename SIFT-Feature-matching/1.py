# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate ORB detector
# orb = cv.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)


# # create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate SIFT detector
# sift = cv.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()











# Reference: https://github.com/pulkit15158/SIFT-Matching/blob/master/A2.py

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math




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



def my_Dlt_algo(src_pts,dst_pts):
	a = np.zeros((2*len(src_pts),9))
	for i in range(len(src_pts)):
		a[2*i] = [-src_pts[i][0],-src_pts[i][1],-1,0,0,0,dst_pts[i][0]*src_pts[i][0],dst_pts[i][0]*src_pts[i][1],dst_pts[i][0]]
		a[2*i+1] = [0,0,0,-src_pts[i][0],-src_pts[i][1],-1,dst_pts[i][1]*src_pts[i][0],dst_pts[i][1]*src_pts[i][1],dst_pts[i][1]]
	u,s,v = np.linalg.svd(a)
	h = v[8].reshape((3,3))
	s = np.linalg.svd(a)
	return h/h[2,2]	


MIN_MATCH_COUNT = 10

img1 = cv2.imread('Q3/test2.jpeg', 0)          # queryImage
img2 = cv2.imread('Q3/collage.jpg', 0) # trainImage

# Initiate SIFT detector
# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)



# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     print M
#     print mask 
#     matchesMask = mask.ravel().tolist()

#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)

#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    print M 
    print mask 

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None









# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 0)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

# # draw_params = dict(matchColor = (0,255,0),
# #                    singlePointColor = (255,0,0),
# #                    matchesMask = matchesMask,
# #                    flags = 0)

# # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (255,0,0),
                   # matchesMask = matchesMask, # draw only inliers
                   flags = 0)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
