# ---------------------------------------------------------------------
# Code for running SURF and then getting the matches by useing knn.
# The matches are used in RANSAC, in order to find a homography
# and inliers.
#
# Alexander Karlsson
# ---------------------------------------------------------------------
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from ransac import *
import blender

def matches(img1,img2):
	hessian_tresh = 500 # may speed up, but gives smaller amount of kp.
	surf = cv2.SURF(500,upright=True)
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
	
	H,inliers,inliers2 = ransac(img1p,img2p,30,4.0)
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


from PIL import Image, ImageFilter, ImageOps, ImageChops
# Test code	
img1 = cv2.imread('bilds/book2.jpg')
img2 = cv2.imread('bilds/book1.jpg') 
img1 = img1[:, :, ::-1]
img2 = img2[:, :, ::-1]

img1_pts, img2_pts, inliers2, inliers1,H = matches(img1,img2)
print '----------------'

#plot_inliers(img1,img1_pts,inliers1,img2,img2_pts,inliers2)

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]
dim = array((h2,w2))*matrix([[H[0][0],H[0][1]],[H[1][0],H[1][1]]])
dim = dim.tolist()
dim1 = array((h1,w1))*matrix([[H[0][0],H[0][1]],[H[1][0],H[1][1]]])
dim1 = dim1.tolist()
h = int(round(dim[0][1]*1.2))
w = w2 + int(round(dim[0][0]*3.5))

img_2 = cv2.warpPerspective(img2,H,(w,h))
#affine transformation matrix, the picture should align if H is ok
mat = matrix([[1, 0, 0], [0, 1, 0]], dtype=float)
im2w = cv2.warpAffine(img_2,mat,(w,h))
im1w = cv2.warpAffine(img1,mat,(w,h))
im2w = cv2.cvtColor(im2w,cv2.COLOR_BGR2RGB)
im1w = cv2.cvtColor(im1w,cv2.COLOR_BGR2RGB)
#D = cv2.max(im2w,im1w)
#ima2 = cv2.cvtColor(D,cv2.COLOR_BGR2RGB)
#ima1 = ImageOps.flip(Image.fromarray(ima2))
#ima1 = ima1.filter(ImageFilter.SMOOTH)

#ima = ImageOps.flip(Image.fromarray(im1w))
#imb = ImageOps.flip(Image.fromarray(im2w))
#test = ImageChops.lighter(ima,imb)


#test = test.filter(ImageFilter.SMOOTH)

A = cv2.cvtColor(im1w,cv2.COLOR_RGB2GRAY)
B = cv2.cvtColor(im2w,cv2.COLOR_RGB2GRAY)
#A = im1w
#B = im2w
#im1 = cv2.max(A,B)
im1 = blender.find_seam(A,B,w2)

#cv2.imwrite('result_books.jpg',ima2)
plt.imshow(im1)
plt.show()


