#!/usr/bin/env python
"""--------------------------------------------------------
 Functions for stitching images.
 Calls seam_finder.py and then performs alpha-blending

 Alexander Karlsson, 2015
--------------------------------------------------------"""

import numpy as np
import cv2 
import seam_finder as seamf		


def stitch_horizontal(A,B,source):
	""" Stitches two images that should be aligned horizontally """

	row,col = A.shape[:2]
	seam = seamf.find_vertical_seam(A,B,source)
	for p in seam:
		# Add weighted for smoother transiion
		weight = 0
		dist = 0
		for x in xrange(0,5):
			weight += 0.2
			dist += 5

			A[p[1],p[0]-dist:p[0]] = cv2.addWeighted(A[p[1],p[0]-dist:p[0]],
			weight,B[p[1],p[0]-dist:p[0]],1-weight,0.5)
			B[p[1],p[0]:p[0]+dist] = cv2.addWeighted(B[p[1],p[0]:p[0]+dist],
			weight,B[p[1],p[0]:p[0]+dist],1-weight,0.5)

		# Set pixels "after" seam to zero
		A[p[1],p[0]:col-1] = 0
		B[p[1],0:p[0]] = 0
		
	C = cv2.add(A,B)
	return C


def stitch_vertical(A,B,source):
	""" Stitches two images that should be aligned vertically """

	row,col = A.shape[:2]
	seam = seamf.find_horizontal_seam(A,B,source)
	for p in seam:
		# Add weighted for smoother transition
		weight = 0
		dist = 0
		for x in xrange(0,5):
			weight += 0.2
			dist += 5

			A[p[1]-dist:p[1],p[0]] = cv2.addWeighted(A[p[1]-dist:p[1],p[0]],
			weight,B[p[1]-dist:p[1],p[0]],1-weight,0.5)
			B[p[1]:p[1]+dist,p[0]] = cv2.addWeighted(B[p[1]:p[1]+dist,p[0]],
			weight,B[p[1]:p[1]+dist,p[0]],1-weight,0.5)

		# Set pixels "after" seam to zero
		A[p[1]:row-1,p[0]] = 0
		B[0:p[1],p[0]] = 0
		
	C = cv2.add(A,B)
	return C

