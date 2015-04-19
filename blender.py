import numpy as np,sys
import cv2 
from matplotlib import pyplot as plt
import time

# Blends two cv2 images using a simple method.
# Should work good on outdoor photage. 
def poor_mans_blending(img1, img2):
	return  cv2.max(img1,img2)


# Finds a seam between A and B, 
# Possible jumps in ths algorithm, i,j is the current
# pixel
#+------+-------+-------+
#|	 |  i,j	  |	  |	|	
#|_______|________|_______|	|
#|i+1,j-1| i+1,j  |i+1,j+1|	|
#|_______|________|_______|	V
#
def find_seam(A,B,source):
	#d_sq = (cv2.min(A,B))**2
	d_sq = (A-B)**2
	cv2.imshow('Image',d_sq)
	cv2.waitKey()

	row,col = A.shape[:2]
	D = d_sq	
	pts = D.copy()
	D[row-1,col-1] = d_sq[row-1,col-1]
	points = []
	i = 0
	j = source/2
	print 'h ', row, ' w ', col

	while j < col-2 and i < row-2:
		D[:,j+1] = d_sq[:,j+1]
		print 'j' ,j, ' ', col
		print 'i' ,i, ' ', row	
		D[i+1,] = d_sq[i+1,]		

#		a = D[i+1,j]
#		b = D[i-1,j+1]
#		c = D[i,j+1]
#		d = D[i+1,j+1]

		#a = D[i,j+1]
		b = D[i+1,j+1]
		c = D[i+1,j]
		d = D[i+1,j-1]
		#e = D[i,j-1]

		min_term = 255
		#if a<=b and a <= c and a <= d:
		#	j += 1
		#	min_term = a

		if  b <= c and b <= d:
			i += 1
			j += 1
			min_term = b

		elif c <= b and c <= d:
			i += 1
			min_term = c

		elif d <= b and d <= c:
			j -= 1
			i += 1
			min_term = d


		D[i][j] = d_sq[i][j] + min_term		
		pts[i][j] = 255
		points.append([j,i])
		if j > 300:
			cv2.imshow('Image',pts)
			cv2.waitKey()
		
	
	for p in points:			
		plt.plot(p[0],p[1],'mo')
	plt.show()	
	return D

