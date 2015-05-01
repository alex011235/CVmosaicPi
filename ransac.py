#!/usr/bin/env python
# ---------------------------------------------------------------------
# RANSAC implemented in python.
#
# Alexander Karlsson
# ---------------------------------------------------------------------
from numpy import *
import scipy
import scipy.linalg

def pflat(x):
	""" Divides each row with the third row """

	x = x[0:3] / x[2]
	return x 


def homogeneous(x):
	""" Converts to homogeneous cordinates """

	return vstack((x,ones((1,x.shape[1]))))


def homography(x1,x2):
	""" Computes the homography between x1 and x2 using DLT and SVD """

	N = x1.shape[1]
	M = zeros((2*N,9))

	# DLT
	for i in range(N):
		a = x1[0][i]; b = x1[1][i]; c = 1
		A = x2[0][i]; B = x2[1][i]

		M[2*i] = [-a, -b, -c, 0,0,0, A*a, A*b, A]
		M[2*i+1] = [0,0,0, -a, -b, -c, B*a, B*b, B]

	H = linalg.svd(M)[2][8].reshape((3,3))
	return H / H[2,2]


def ransac(xA, xB, max_iter = 50, err = 5):
	""" RANSAC algorithm, finds the homography from xA to xB. 
	Stops when max_iter is reached. err shouldbe about 5 pxls. """

	inliers_record = 0 
	xa = xA[0] 
	ya = xA[1]
	xb = xB[0]
	yb = xB[1]
	H_best = []
	best_inliers1 = []
	best_inliers2 = []	

	for i in range(0, max_iter):
		xA_rand = zeros((4,2))
		xB_rand = zeros((4,2))

		# Choose four random points
		for j in xrange(0, 4):
			rand_index = random.randint(xA.shape[1])
			xA_rand[j,0] = xa[rand_index]; 
       			xA_rand[j,1] = ya[rand_index];
        		xB_rand[j,0] = xb[rand_index]; 
        		xB_rand[j,1] = yb[rand_index];
		
		xA_rand = xA_rand.conj().transpose()
		xB_rand = xB_rand.conj().transpose()	
		
		Ht = homography(xB_rand, xA_rand)
		p1 = vstack((xa,ya,ones((1,xa.shape[0]))))
		# New model, better?
		p2 = pflat(linalg.solve(Ht,p1))
		x = p2[0]
		y = p2[1]
		inliers_cand1 = []
		inliers_cand2 = []	
		total = 0
		for k in range(0,len(x)):
			a = array((x[k],y[k])).conj().transpose()
			b = array((xb[k],yb[k])).conj().transpose()
			c = array((xa[k],ya[k])).conj().transpose()
			# compute error
			error = linalg.norm(a-b) # nice!	
			if error < err:
				total += 1
				inliers_cand1.append(a.conj().transpose())
				inliers_cand2.append(c.conj().transpose())
					
		# Check if total exceeds the record, then update the best
		# homography and the inliers.
		if total > inliers_record:
			print total
			inliers_record = total
			H_best = Ht
			best_inliers1 = inliers_cand1
			best_inliers2 = inliers_cand2
	best_inliers1 = array(best_inliers1)
	best_inliers2 = array(best_inliers2)
	H_best = homography(best_inliers1.conj().transpose(),
	best_inliers2.conj().transpose())
	
	return H_best, best_inliers1, best_inliers2
