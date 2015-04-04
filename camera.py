import io
import time
import picamera
from pylab import *
import cv2
import numpy as np

class Camera:
	images = []
	images.append([])	
	col = 0
	camera = picamera.PiCamera()

	def __init__(self):
		time.sleep(1) 	# Warmup time for camera

	def snap_image(self):
		stream = io.BytesIO()	
		self.camera.capture(stream, format='jpeg')
		data = np.fromstring(stream.getvalue(), dtype=np.uint8)
		image = cv2.imdecode(data, 1)
		self.images[self.col].append(image)

	def new_column(self):
		self.images.append([])
		self.col += 1

	def retreive_image_matrix(self):
		return self.images

# Test block

#cam = Camera()
#for i in range(1,3):
#	cam.snap_image()
#	time.sleep(0.5)
#
#cam.new_column()
#
#for i in range(1,3):
#	cam.snap_image()
#	time.sleep(0.5)
#
#
#images = cam.retreive_images()
#
#for imgs in images:
#	for image in imgs:
#
#		image = image[:, :, ::-1] # RGB
#		imshow(image)
#		show()




