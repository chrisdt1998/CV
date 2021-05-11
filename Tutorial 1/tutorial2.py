import numpy as np
import cv2
import matplotlib.pyplot as plt

# Part 1 a

alpha = 0.3
beta = 2
gamma = 1
a = 50
b = 150
y_a = 30
y_b = 200

image1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2a.png', 0)
print(image1.shape)

image2 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2a.png', 0)
image3 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2a.png', 0)

def contrast_stretch(x, a, b, y_a, y_b):
	if x >= 0 and x < a:
		return alpha * x
	elif x >= a and x < b:
		return beta * (x - a) + y_a
	elif x >= b and x < 427:
		return gamma * (x - b) + y_b

def contrast_clip(x, a, b):
	if x >= 0 and x < a:
		return 0
	elif x >= a and x < b:
		return beta * (x - a)
	elif x >= b and x < 427:
		return beta * (b - a)

# for i in range(427):
# 	for j in range(640):
# 		image2[i,j] = contrast_stretch(image2[i, j], a, b, y_a, y_b)
# 		image3[i,j] = contrast_clip(image3[i,j], a, b)
#
"""
cv2.imshow('image', image1)
cv2.waitKey()
cv2.imshow('image', image2)
cv2.waitKey()
cv2.imshow('image', image3)
cv2.waitKey()
"""

# part 1 b
"""
hist = cv2.calcHist([image1],[0],None,[256],[0,256])
plt.hist(hist)
plt.show()

img = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/Unequalized_H.jpg', 0)
plt.hist(img.ravel(), bins=256)
plt.show()
## NEEDS TO BE FINISHED
"""
"""
# part 1 c
image4 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2a.png', 0)
filter = 1/81 * np.ones((9, 9))
padding = 4
sum = 0
for i in range(padding, 427 - padding):
	for j in range(padding, 640 - padding):
		for t in range(-4, 5):
			for s in range(-4, 5):
				sum += 1/81 * image4[i - s, j - t]
		image4[i, j] = sum
		sum = 0

print(image4)
cv2.imshow('image', image1)
cv2.waitKey()
cv2.imshow('image', image4)
cv2.waitKey()
"""

# Part d

# image5 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2a.png', 0)
# filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# padding = 1
# sum = 0
# for i in range(padding, 427 - padding):
# 	for j in range(padding, 640 - padding):
# 		for t in range(-1, 2):
# 			for s in range(-1, 2):
# 				sum += filter[t + 1,s + 1] * image5[i - s, j - t]
# 		image5[i, j] = sum
# 		sum = 0
#
#
# cv2.imshow('image', image1)
# cv2.waitKey()
# cv2.imshow('image', image5)
# cv2.waitKey()

# Part 2 a

img1 = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/lab2b.png', 0)
fft_res = np.fft.fft2(img1)
fft_shift_res = np.fft.fftshift(fft_res)

plt.imshow(np.log10(np.abs(fft_res)), cmap='gray')
plt.show()