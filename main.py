#!/usr/bin/env python
# -------------------------------------------------------------
# Main program
# -------------------------------------------------------------


from servo import *
from camera import *
#from keypoints import *
import cv2
import time
from pylab import *

servo = Servo(2,1)
cam = Camera()
cam.resolution = (600,450)
cam.snap_image()

for i in range(1,4):
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

for i in range(1,4):
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

for i in range(1,4):
	servo.tilt_down()
	servo.stop()
	time.sleep(0.4)
	cam.snap_image()
	


images = cam.retreive_image_matrix()
#keyp = KeyPoints(images, 'fast')
#kp = keyp.keypoints()

for i in range(0,len(images)):
	for j in range(0,len(images[i])):
	
		image = images[i][j]
		#image = image[:, :, ::-1] # RGB
		#keys = kp[i][j]
		#im = cv2.drawKeypoints(image,keys,color=(255,0,255))
		#cv2.imwrite('bilds/test' + str(i)+str(j) + '.jpg',im)
		#imshow(im)
		#show()	






