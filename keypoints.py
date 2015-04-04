#!/usr/bin/env python
# ----------------------------------------------------
# A class for extracting keypoints from an image.
# One can choose 'fast', 'sift' and 'surf'. Although
# 'fast' is better for the Pi, since it is fast.
#
# Alexander Karlsson, 2015
# ----------------------------------------------------
 
import numpy as np
import cv2

class KeyPoints:
	keypoints = []
	keypoints.append([])
	
	# images is an {m x n} matrix, where each entry is 
	# image data for every taken image
	def __init__(self,images, method = 'fast'):
		self.method = method
		self.images = images


	def keypoints(self):
		if self.method == 'fast':
			return self.fast_keypoints()
		elif self.method == 'sift':
			return self.sift_keypoints()
		elif self.method == 'surf':
			return self.surf_keypoints()

	# FAST
	def fast_keypoints(self):
		fast = cv2.FastFeatureDetector()
		col = 0	
		keyp = []
		keyp.append([])

		for imgs in self.images:
			for img in imgs:			
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			
				kp = fast.detect(gray,None)
				keyp[col].append(kp)
			
			keyp.append([])
			col += 1

		return keyp

	# SLOWEST
	def sift_keypoints(self):
		surf = cv2.SIFT()
		col = 0
		keyp = []
		keyp.append([])
		
		for imgs in self.images:
			for img in imgs:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				kp = surf.detect(gray,None)
				keyp[col].append(kp)

			keyp.append([])
			col += 1

		return keyp

	# SLOW
	def surf_keypoints(self):
		sift = cv2.SURF()
		col = 0
		keyp = []
		keyp.append([])
		
		for imgs in self.images:
			for img in imgs:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				kp = sift.detect(gray,None)
				keyp[col].append(kp)

			keyp.append([])
			col += 1

		return keyp


