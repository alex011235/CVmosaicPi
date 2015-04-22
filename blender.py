import numpy as np
import cv2 
from matplotlib import pyplot as plt
import scipy
import scipy.linalg

# Blends two cv2 images using a simple method.
# Should work good on outdoor photage. 
def poor_mans_blend(img1, img2):
	return  cv2.max(img1,img2)


# Finds a seam between A and B, 
# Possible jumps in ths algorithm, (i,j) is the current
# pixel:
#+-------+--------+-------+
#|	 |  i,j	  |  	  |	|	
#|_______|________|_______|   +-|-+
#|i+1,j-1| i+1,j  |i+1,j+1|   | | |
#|_______|________|_______|   V	V V
#
def _find_vertical_seam(A,B,source):
	A = cv2.cvtColor(A,cv2.COLOR_RGB2GRAY)
	B = cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)	
	d_sq = (A-B)**2
	row,col = A.shape[:2]	
	points = []
	
	i = 0
	j = source/2
	while j < col-2 and i < row-2:
		b = d_sq[i+1,j+1]
		c = d_sq[i+1,j]
		d = d_sq[i+1,j-1]

		if  b <= c and b <= d:
			i += 1
			j += 1
		elif c <= b and c <= d:
			i += 1
		elif d <= b and d <= c:
			j -= 1
			i += 1

		points.append([j,i])

	return np.array(points)


def stitch(A,B,source):
	row,col = A.shape[:2]
	seam = _find_vertical_seam(A,B,source)
	mean = []
	for p in seam:
		# Add weighted for smoother blending
		weight = 0
		dist = 0
		for x in xrange(0,10):
			weight += 0.1
			dist += 5

			A[p[1],p[0]-dist:p[0]] = cv2.addWeighted(A[p[1],p[0]-dist:p[0]],
			weight,B[p[1],p[0]-dist:p[0]],1-weight,0.5)
			B[p[1],p[0]:p[0]+dist] = cv2.addWeighted(B[p[1],p[0]:p[0]+dist],
			weight,B[p[1],p[0]:p[0]+dist],1-weight,0.5)


		# Set pixels "after" seam to zero
		A[p[1],p[0]:col-1] = 0
		B[p[1],0:p[0]] = 0
		# Stitch the images at the seam using the mean value 
		mean.append(A[p[1],p[0]-1:p[0]])
		mean.append(B[p[1],p[0]:p[0]+1])
		M = [0,0,0]
		for sub in mean:
			
			for m in sub:
				M += m 		
				
		M /= 2
				
		A[p[1],p[0]-1:p[0]] = M
		B[p[1],p[0]:p[0]+1] = M
		
		mean = []


	C = cv2.add(A,B)
	plt.imshow(C)
	plt.show()
	C = C[:, :, ::-1]
	cv2.imwrite('stitch_after_seam_found.jpg',C)

	#cv2.imshow('Image', C)
	#cv2.waitKey()
	return C
