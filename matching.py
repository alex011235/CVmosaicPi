# ---------------------------------------------------------------------
# Code for running SURF and then getting the matches by useing knn.
# The matches are used in RANSAC, in order to find a homography
# and inliers.
#
# Alexander Karlsson
# ---------------------------------------------------------------------
import numpy as np
import cv2 
import cv
from matplotlib import pyplot as plt
from ransac import *

def matches(img1,img2):
	hessian_tresh = 500 # may speed up, but gives smaller amount of kp.
	surf = cv2.SURF(300,upright=True)
	# SURF keypoints and descriptors
	kp1, des1 = surf.detectAndCompute(img1,None)
	kp2, des2 = surf.detectAndCompute(img2,None)
	# Find matches using k nearest neighbord
	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks = 10)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# best_matches. 
	best_m = [] 
	for a,b in matches:
	    if a.distance < 0.7*b.distance:
		best_m.append(a)

	img1_pts = np.float32([ kp1[m.queryIdx].pt for m in best_m ]).reshape(-1,1,2)	
	img2_pts = np.float32([ kp2[m.trainIdx].pt for m in best_m ]).reshape(-1,1,2)

	w = len(best_m)
	img1p = zeros((2,w))
	img2p = zeros((2,w))

	# Get some useful data
	i = 0;
	for m in img1_pts:
		img1p[0][i] = m[0][0]
		img1p[1][i] = m[0][1]
		i += 1		

	i = 0;
	for m in img2_pts:
		img2p[0][i] = m[0][0]
		img2p[1][i] = m[0][1]
		i += 1		
	
	H,inliers,inliers2 = ransac(img1p,img2p,40,3.0)
	return img1_pts, img2_pts, inliers, inliers2, H


# Very slow
def plot_inliers(img1, img1pts, inliers1, img2, img2pts, inliers2):
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	ax1.plot(inliers1[0][0],inliers1[0][1],'go')
	ax1.imshow(img1)	
	for p in img1pts:      		
		ax1.plot(p[0][0],p[0][1],'ro')

	for p in inliers1:
		ax1.plot(p[0],p[1],'go')

	ax2.imshow(img2)
	ax2.plot(inliers2[0][0],inliers2[0][1],'go')
	
	for p in img2pts:
		a, = ax2.plot(p[0][0],p[0][1],'ro')
		ax2.legend('fhdjk')

	for p in inliers2:
		b, = ax2.plot(p[0],p[1],'go')

	ax2.legend([a, b], ["SURF matches", "RANSAC inliers"])
	plt.show()



# Test code	
img1 = cv2.imread('bilds/book2.jpg',0)
img2 = cv2.imread('bilds/book1.jpg',0) 
img11 = cv2.imread('bilds/book2.jpg')
img11 = img11[:, :, ::-1]
img22 = cv2.imread('bilds/book1.jpg')
img22 = img22[:, :, ::-1]

img1_pts, img2_pts, inliers2, inliers1,H = matches(img1,img2)
print '----------------'
print inliers1[0]
print inliers2[0]

#plot_inliers(img11,img1_pts,inliers1,img22,img2_pts,inliers2)

img_2 = cv2.warpPerspective(img22,H,(1000,750))
#affine transformation matrix
mat = matrix([[1, 0, abs(inliers1[0][1]-inliers2[0][1])],[0, 1, abs(inliers1[0][0]-inliers2[0][0])]])
mat1 = matrix([[1, 0, 0],[0, 1, 0]])
img_2_warped = cv2.warpAffine(img_2,mat,(1500,1100))
img_1_warped = cv2.warpAffine(img11,mat,(1500,1100))

b = cv2.addWeighted(img_1_warped,0.5,img_2_warped,0.5,0.5)
plt.imshow(b)
plt.show()

