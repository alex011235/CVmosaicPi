#!/usr/bin/env python
"""--------------------------------------------------------
 The functions in this file will hopefully find the 
 seam between two images A and B. 
 If failure, try to change the source.

 Alexander Karlsson, 2015
 --------------------------------------------------------"""
import cv2
import numpy as np

def find_vertical_seam(A,B,source):
	""" Finds the vertical seam between the images A and B. """

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


def find_horizontal_seam(A,B,source):
	""" Finds the horizontal seam between the images A and B. """
	
	A = cv2.cvtColor(A,cv2.COLOR_RGB2GRAY)
	B = cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
	d_sq = (A-B)**2
	row,col = A.shape[:2]
	points = []

	i = source/2
	j = 0
	while j < col-2 and i < row-2:
		b = d_sq[i-1,j+1]
		c = d_sq[i,j+1]
		d = d_sq[i+1,j+1]

		if  b <= c and b <= d:
			i -= 1
			j += 1
		elif c <= b and c <= d:
			j += 1
		elif d <= b and d <= c:
			j += 1
			i += 1

		points.append([j,i])

	return np.array(points)
