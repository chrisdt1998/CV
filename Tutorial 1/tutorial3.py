import cv2
import numpy as np

# Part a
c = 1
lena = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/Lena.jpg')
img_lena1 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)/255
img_lena2 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)/255
filer_x = np.array([[-1, 0, 1], [-c, 0, c], [-1, 0, 1]])
filer_y = np.array([[-1, -c, -1], [0, 0, 0], [1, c, 1]])

def convolution(img, kernel, padding=True):
    """ Performs convolution operation given an image and a kernel

        Parameters
        ----------
        img : array_like
        1-channel image
        kernel : array-like
        kernel (filter) for convolution

        Returns
        -------
        np.ndarray
        result of the convolution operation
    """
    result = np.zeros_like(img)
    p_size_i = kernel.shape[0] // 2
    p_size_j = kernel.shape[1] // 2

    if padding:
        padded_img = np.zeros((img.shape[0] + 2 * p_size_i, img.shape[1] + 2 * p_size_j))
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1
        padded_img[i_first: i_last + 1, j_first: j_last + 1] = img
    else:
        padded_img = img.copy()
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1

    for i in range(i_first, i_last):
        for j in range(j_first, j_last):
            window = padded_img[i - p_size_i: i + p_size_i + 1, j - p_size_j: j + p_size_j + 1]
            res_pix = np.sum(window * kernel)
            result[i - p_size_i, j - p_size_j] = res_pix
    return result

image_mx = convolution(img_lena1, filer_x, False)
image_my = convolution(img_lena2, filer_y, False)
image_final = np.sqrt(np.power(image_mx, 2) + np.power(image_my, 2))
thresh = 0.05
image_final = cv2.threshold(image_final, thresh, 255, cv2.THRESH_BINARY)[1]
print(image_final)
cv2.imshow('image', image_final)
cv2.waitKey()


# Part b
"""
# Part c
from scipy.io import loadmat

log5 = loadmat('./Log5.mat')['Log5']
log17 = loadmat('./Log17.mat')['Log17']

laplacian = np.array([[0, 1, 1], [1, -4, 1], [0, 1, 0]])
def convolution(filter, image):
    padding = 1
    sum = 0
    image_dim = image.shape
    for i in range(padding, image_dim[0] - padding):
        for j in range(padding, image_dim[1] - padding):
            for t in range(-1, 2):
                for s in range(-1, 2):
                    sum += filter[t + 1, s + 1] * image[i - s, j - t]
            image[i, j] = sum
            sum = 0
    return image

lena = cv2.imread(r'/Users/chris/Documents/GitHub/CV/Tutorial 1/Lena.jpg')
img_lena1 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)/255
"""