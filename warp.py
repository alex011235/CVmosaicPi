import os
import cv2
import numpy as np


# Warps an image to a spherical image
# Input: img: source image, [k1,k2]: radial distortion paramters
# f: focallength
def spherical_warp(img, f, k1=-0.21, k2=0.26):

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
	borderMode=cv2.BORDER_DEFAULT)
	
	return warped

'''
from matplotlib import pyplot as plt
f = 589.645443005 # calibrated
img1 = cv2.imread('bilds/book2.jpg')
img2 = spherical_warp(img1,f)
img2 = img2[:, :, ::-1]
plt.imshow(img2)
plt.show()
#cv2.imshow('Spherical',img2)
#cv2.waitKey() '''
