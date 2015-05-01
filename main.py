#!/usr/bin/env python
# -------------------------------------------------------------
# Main program
# -------------------------------------------------------------
from servo import *
from camera import *
import cv2
import time
from pylab import *
import warp
import matching
import stitcher

def camera_program(x):
	""" Snaps x images horizontally and y vertically."""
	
	servo = Servo(2,1)
	cam = Camera()

	for i in xrange(x): #horizontal
		time.sleep(1)
		cam.snap_image()
		time.sleep(1)
		servo.move_right()

		
	return cam.retreive_image_matrix()
			

def stitch_horizontal(img1,img2):
	f = 581.94748361
	#f = 597.0
	img1 = warp.spherical_warp(img1, f)
	img2 = warp.spherical_warp(img2, f)
	# Call matches for extracting inliers and homography
	H = matching.matches(img1,img2)
	# Try to calculate the stitched image size
	h2,w2 = img2.shape[:2]
	tx = H[0,2]
	ty = H[1,2]
	h = int(round(h2 + ty)*1.05)+500
	w = int(round(w2 + tx)*1.09)+500
	H = np.matrix([[1,0,tx],[0,1,0],[0,0,1]])
	img2_warped = cv2.warpPerspective(img2,H,(w,h))
	#affine transformation matrix, the picture should align if H is ok
	mat = matrix([[1, 0, 0], [0, 1, 0]], dtype=float)
	im1w = cv2.warpAffine(img1,mat,(w,h))
	im2w = cv2.warpAffine(img2_warped,mat,(w,h))
	# Find a seam between the two images
	A = stitcher.stitch_horizontal(im1w,im2w,w2+200)

	return A




images = camera_program(2)

for i in range(0,len(images)):
	j = 0
	while j < len(images[i])-1:
	
		img1 = images[i][j]
		img2 = images[i][j+1]
		img1 = img1[:, :, ::-1]
		img2 = img2[:, :, ::-1]
		
		
		im_stitch = stitch_horizontal(img1,img2)	
		
		#image = image[:, :, ::-1] # RGB
		#cv2.imwrite('bilds/test' + str(i)+str(j) + '.jpg',im)
		imshow(im_stitch)
		show()	
		
		j += 1



'''

cam.resolution = (600,450)
cam.snap_image()

for i in range(0,1):
	servo.tilt_down()
	servo.stop()
	time.sleep(0.4)
	cam.snap_image()
		

servo.stop()	
servo.move_left()
servo.stop()	
time.sleep(0.5)
cam.snap_image()
time.sleep(0.5)

for i in range(0,1):
	servo.tilt_up()
	servo.stop()
	time.sleep(0.4)
	cam.snap_image()


servo.stop()	
time.sleep(0.5)
servo.move_left()
servo.stop()
cam.snap_image()
time.sleep(0.5)

for i in range(0,1):
	servo.tilt_down()
	servo.stop()
	time.sleep(0.4)
	cam.snap_image()
	


images = cam.retreive_image_matrix()

#for i in range(0,len(images)):
#	for j in range(0,len(images[i])):
#	
#		image = images[i][j]
#		image = image[:, :, ::-1] # RGB
#		#cv2.imwrite('bilds/test' + str(i)+str(j) + '.jpg',im)
#		imshow(image)
#		show()	



'''

