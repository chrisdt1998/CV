import numpy as np
import cv2
import matplotlib.pyplot as plt

# part b
def partb():
	x_cords = np.random.random(100)
	y_cords = x_cords*x_cords
	A = np.tile(y_cords,(100,1))
	print(A)
	plt.imshow(A)
	plt.show()

# part c
def partc():
	image1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab1a.png')
	cv2.imshow('image', image1)
	cv2.waitKey()

# part d
def partd():
	image1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab1a.png', 0)
	cv2.imshow('image', image1)
	cv2.waitKey()

# part e
def parte():
	image1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab1a.png')
	hsvImage = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	fig = plt.figure()
	for i in range(1,4):
		img = hsvImage[:, :, i-1]
		fig.add_subplot(3,1,i)
		plt.imshow(img)
	plt.show()
	"""
	cv2.imshow('Original image',image1)
	cv2.imshow('HSV image', hsvImage)
	cv2.waitKey()
	"""

# part f
def partf():
	BWimage = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab1a.png', 0)
	image1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab1a.png')
	fig = plt.figure()
	for i in [0, 50, 100, 150, 200, 255]:
		ret, BWimage = cv2.threshold(image1, i,255, cv2.THRESH_BINARY)
		fig.add_subplot(3,2, [0, 50, 100, 150, 200, 255].index(i)+1)
		plt.imshow(BWimage)
	plt.show()

# part g
def partg():
	pass



partf()







