# Algorithms in Computer Vision

The repository contains the implementation of various computer vision algorithms from scratch. 

## Introduction
- This repository contains python implementation of the following algorithms from scratch. 


  i. CNN implementation using PyTorch. 
  
  ii. Classification of Cifar dataset using Alexnet pre-trained model. 
  
  iii. Segmentation using the following clustering algorithms: K-means, Seeding, mean-shift. 
  
  iv. Camera Clibration using OpenCV
  
  v. Implementation of Circular Hough Transformation 
  
  vi. Corner Detection 
  
  vii. Find matching regions in a image using SIFT features. 
  
  viii. Classification of CIFAR dataset using visual bag of words. 
  
  ix. Filtering algorithms: Gaussian, Median, etc. 
  
  x. Finding disparity maps using OpenCV
  
  xi. Image (hidden) Watermarking algorithm 
  
  xii. Image stitching to create a panorama. 

## Implementation details and Results 


### Clutering 

#### K-means clutering 


##### With a 3D (R,G,B) feature vector  for different number of classes

![Original Image](/Assets/k-mean-1.png)



##### With a 5D (R,G,B,u,v) feature vector  for different number of classes

![Original Image](/Assets/k-mean-2.png)


#### Seeding clustering and Thesholding Algorithms for Image Segmentation 

![Original Image](/Assets/seeding-1.png)


![Original Image](/Assets/seeding-2.png)



#### Mean-Shift Clustering 

Bin_seeding was set to TRUE to reduce the computational complexity. The quartile value for output image 1 was set to 0.25. Estimate bandwidth function was used to find the exact bandwidth to be used. The quartile value was grid searched. Moreover the higher quartile brings the value closer to the median of all pairwise distances. This is not required. Thus it was reduced from the standard value of 0.3. 

For output image 2, the bandwidth was manually set to 2. This gave a very large number of cluster centers, in other terms it gave a large number of image color classes. Which is the reason behind it looking similar to the original image. 



![Original Image](/Assets/mean-shift-1.png)

![Original Image](/Assets/mean-shift-2.png)



### Circular Hough Transform 

The first step is resizing the image to 1/4th of its original size keeping the aspect ratio the same. The is done to speed up the algorithm. The second step is using a gaussian filter and the a canny edge detector on the image. Which finds the edges in the image. Circular hough transform is used with a dynamic threshold for every radius value. The hough transform is only used on locations where an edge is detected. Using this dynamic thresholding approach I was able to detect the inner circles as well.

![Original Image](/Circular-Hough-transform/Q1.jpeg)

Figure 1: Original Input Image 

![Filtered Image](/Circular-Hough-transform/filtered.jpg?raw=true)

Figure 2: Image is reduced to 1/8th its original size and filtered using a gaussian filter to remove noise. 

![Filtered Image](/Circular-Hough-transform/edge.jpg?raw=true)

Figure 3: Edges are detected in the image using canny edge detector 

![Filtered Image](/Circular-Hough-transform/Figure_1-1.png?raw=true)

Figure 4: Output image after performing circular hough transform using a dynamic theshoding for every radius value. 


### Depth Map 

![Original Image](/Assets/depth-map.png)

Both of the images are first projected to a parallel plane system. We calculate the essential matrix for the two images. Now stereo is used to find the depth of each point based on the difference between the two cameras, the focal length. 

### RANSAC and SIFT Feature Matching


![Original Image](/Assets/sift-features.png)

### Building Gaussian Pyramid


![Original Image](/Assets/gaussian-pyramid.png)


![Original Image](/Assets/laplacian-pyramid.png)



### Image Watermarking 


Embedded a watermark of size 1/8th the original image size in the LL subband of second order DWT. The same image was used for the watermarking operation. Implemented [this research paper](https://ieeexplore.ieee.org/document/5560822) for performing the task. 


![Original Image](/Assets/watermarking.png)




### Filtering Algorithms 


#### Average Filtering

![Original Image](/Assets/avg-filter.png)


#### Gaussian Filtering

![Original Image](/Assets/gaussian-filter.png)


#### Median Filtering

![Original Image](/Assets/median-filtering.png)

![Original Image](/Assets/median-1.png)




### Image Stitching 

![Original Image](/Assets/image-stitching.png)

Used SIFT to find the keypoint matches between the images. Using these keypoint matches, the homography was calculated. And the images were added together using this homography matrix through which both the images were stitched together to the first one by converting it to the first.   


### Corner Detection 


![Original Image](/Assets/corner.png)


Upon rotation the same threshold gave good results however upon compression by a factor of 2 using bilinear interpolation shows poorer results on the same threshold value. This is because of lower resolution of the image upon compression. Upon increasing the threshold lesser number of points are detected and while lowering the threshold more points are detected and more edges are also classified as a corner. 



