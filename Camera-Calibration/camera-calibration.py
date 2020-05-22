import numpy as np
import cv2
import glob


### Code referred from the official OpenCV documentation ############



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((12*12,3), np.float32)
objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 


for filename in glob.glob('Camera Calibration/*.bmp'):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (12,12),None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)




print len(objpoints), len(imgpoints), gray.shape

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print 'ret', ret
print 'Camera Matrix', mtx
print 'Distortion Array', dist
print 'Rotation Matrix', rvecs
print 'Translation Matrix', tvecs


apertureSize = 1
apertureHeight = 1

fovx, fovy, focalLength, principalPoint,aspectRatio = cv2.calibrationMatrixValues(mtx, gray.shape, apertureSize, apertureHeight)


print 'Fovx', fovx
print 'Fovy', fovy
print 'Focal Length', focalLength
print 'Principal point', principalPoint
print 'aspect ratio', aspectRatio

img = cv2.imread('Camera Calibration/Left12.bmp')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error: ", mean_error/len(objpoints)

 
