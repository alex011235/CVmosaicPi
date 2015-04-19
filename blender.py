import numpy as np
import cv2 
from matplotlib import pyplot as plt

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

	for p in seam:
		A[p[1],p[0]:col-1] = 0
		B[p[1],0:p[0]] = 0

	C = cv2.add(A,B)
	plt.imshow(C)
	plt.show()
	C = C[:, :, ::-1]
	cv2.imwrite('stitch_after_seam_found.jpg',C)
	#cv2.imshow('Image', C)
	#cv2.waitKey()








