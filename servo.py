#!/usr/bin/env python
""" -------------------------------------------------------------
 A class for controlling the servos. This must have the 
 'Adafruit_PWM_Servo_Driver' in the current directory.
 
 Alexander Karlsson, 2015
 -------------------------------------------------------------"""

from Adafruit_PWM_Servo_Driver import PWM
import time

class Servo:
	""" The functions are used to move the camera just about 
	so that two images aligns with at least 50% overlap."""

	pwm = PWM(0x40)
	
	def __init__(self,h_channel, v_channel):
		self.hor = h_channel
		self.ver = v_channel


	def move_left(self):
		self.pwm.setPWM(self.hor,4000,270)
		time.sleep(0.005)
		self.stop()

	def move_right(self):
		self.pwm.setPWM(self.hor,4000,3000)
		time.sleep(0.00003)
		self.stop()

	def tilt_up(self):
		self.pwm.setPWM(self.ver,4000,270)
		time.sleep(0.01222)
		self.stop()

	def tilt_down(self):
		self.pwm.setPWM(self.ver,500,0)
		time.sleep(0.0005)
		self.stop()			

	def stop(self):
		self.setServoPulse(self.ver,0)
		self.setServoPulse(self.hor,0)

	def setServoPulse(self, channel, pulse):
		""" Useful when to stop the servos """

		pulseLength = 1000000                   # 1,000,000 us per second
		pulseLength /= 60                       # 60 Hz
		pulseLength /= 4096                     # 12 bits of resolution
		pulse *= 1000
		pulse /= pulseLength
		self.pwm.setPWM(channel, 0, pulse)		

