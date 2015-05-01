#!/usr/bin/env python
""" -------------------------------------------------------------
 Class for controlling the Raspberry Pi camera. This class 
 contains functionality for snapping images and storing them 
 in a matrix. 

 Alexander Karlsson, 2015
 -------------------------------------------------------------"""
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
	camera.resolution = (640,450)

	def __init__(self):
		time.sleep(2) 	# Warmup time for camera


	def snap_image(self):
		""" Saves an image """

		stream = io.BytesIO()	
		self.camera.capture(stream, format='jpeg')
		data = np.fromstring(stream.getvalue(), dtype=np.uint8)
		image = cv2.imdecode(data, 1)
		self.images[self.col].append(image)


	def new_row(self):
		self.images.append([])
		self.col += 1


	def retreive_image_matrix(self):
		""" Returns all images """

		return self.images

