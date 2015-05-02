#!/usr/bin/env python
"""--------------------------------------------------------
 Transforms image coordinates to spherical coordinates.

 Alexander Karlsson, 2015
--------------------------------------------------------"""
import os
import cv2
import numpy as np

def spherical_warp(img, f):
	""" Spherical warp of the image img. """

	#k1 = 0.8
	#k2 = 0.8
	k1 =  0.63151103
	k2 = -0.03377441
	img_shape = np.array(img.shape)
	# calculate minimum y value
	min_y = np.sin(0.0)
	# Get some spherical coordinates
	# The spehere is  parameterized by the angles (theta,phi)
	one = np.ones((img_shape[0],img_shape[1]))
	theta = one * np.arange(img_shape[1])
	phi = one.T * np.arange(img_shape[0])
	phi = phi.T
	
	theta = ((theta - 0.5 * img_shape[1]) / f)
	phi = ((phi - 0.5 * img_shape[0]) / f - min_y)
	xt = np.sin(theta) * np.cos(phi)
	yt = np.sin(phi)
	zt = np.cos(theta) * np.cos(phi)
	
	eucl = [xt,yt,zt]
	eucl[0] = eucl[0]/eucl[2]
	eucl[2] = eucl[1]/eucl[2]
	zt = 1.0

	r_sq = xt**2 + yt**2
	xt = xt * (1 + k1*r_sq + k2*r_sq**2)
	yt = yt * (1 + k1*r_sq + k2*r_sq**2)
	
	xn = 0.5 * img_shape[1] + xt * f
	yn = 0.5 * img_shape[0] + yt * f

	uvImg = np.dstack((xn,yn))

	# warp image 
	h,w = img.shape[:2]
	mask = cv2.inRange(uvImg[:,:,1],0,h-1)&cv2.inRange(uvImg[:,:,0],0,w-1)
	warped = cv2.remap(img, uvImg[:, :, 0].astype(np.float32),\
	     uvImg[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, 
	borderMode=cv2.BORDER_TRANSPARENT)
	return warped

