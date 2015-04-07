import numpy as np
import cv2
from matplotlib import pyplot as plt

def matches(img1,img2):
	hessian_tresh = 500 # may speed up, but gives smaller amount of kp.
	surf = cv2.SURF(500,upright=True)
	# SURF keypoints and descriptors
	kp1, des1 = surf.detectAndCompute(img1,None)
	kp2, des2 = surf.detectAndCompute(img2,None)
	# Find matches using k nearest neighbor
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
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
	return img1_pts, img2_pts

# Test code	
img1 = cv2.imread('bilds/ny2.jpg',0)
img2 = cv2.imread('bilds/ny1.jpg',0) 
img11 = cv2.imread('bilds/ny2.jpg')
img11 = img11[:, :, ::-1]
img22 = cv2.imread('bilds/ny1.jpg')
img22 = img22[:, :, ::-1]

img1_pts, img2_pts = matches(img1,img2)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
 
ax1.imshow(img11)
for p in img1_pts:      		
	ax1.plot(p[0][0],p[0][1],'ro')

ax2.imshow(img22)
for p in img2_pts:
	ax2.plot(p[0][0],p[0][1],'ro')

plt.show()

