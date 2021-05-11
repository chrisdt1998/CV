import scipy.io as sio
from scipy import spatial
from skimage import color
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def plotclusters3D(data, labels, peaks):
	"""
	Plots the modes of the given image data in 3D by coloring each pixel
	according to its corresponding peak.

	Args:
		data: image data in the format [number of pixels]x[feature vector].
		labels: a list of labels, one for each pixel.
		peaks: a list of vectors, whose first three components can
		be interpreted as BGR values.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
	rgb_peaks = bgr_peaks[...,::-1]
	rgb_peaks /= 255.0
	for idx, peak in enumerate(rgb_peaks):
		color = np.random.uniform(0, 1, 3)
		#TODO: instead of random color, you can use peaks when you work on actual images
		# color = peak
		cluster = data[np.where(labels == idx)[0]].T
		ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
	plt.show(block=False)
	plt.pause(10)
	plt.close()

points = sio.loadmat("/Users/chris/Documents/GitHub/CV/Assignment_1_Segmentation/data/pts.mat")["data"].reshape(-1, 3)
#points = color.rgb2lab(points)


class Segmentation():
	def __init__(self, data, r):
		self.data = data
		self.r = r
		self.unedited_data = data.copy()

	def findpeak(self, idx):
		while True:
			# Calculate the euclidean distance between the point at idx and the rest of the points.
			distances = spatial.distance.cdist(np.array(self.data[idx][:]).reshape(1,-1), self.data, metric='euclidean')

			# Find all the coordinates of points where the distance is less than r
			coordinates = np.argwhere(distances < self.r)

			# Array containing all the points which are in the radius
			points_in_radius = self.data[coordinates[:,1]][:]

			# Check if we have reached the threshold and return the peak
			if spatial.distance.cdist(np.array(self.data[idx][:]).reshape(1,-1), np.average(points_in_radius, axis=0).reshape(1,-1), metric='euclidean') < 0.01:
				#print("diff", self.data[idx][:] - np.average(points_in_radius, axis=0))
				self.data[idx][:] = np.average(points_in_radius, axis=0)
				#print("average", np.average(points_in_radius, axis=0))
				#print("data", self.data[idx][:])
				break

			# Compute average of the points in radius and shift window
			self.data[idx][:] = np.average(points_in_radius, axis=0)


	def meanshift(self):
		self.labels = np.zeros(2000)
		for idx in tqdm(range(len(self.data))):
			self.findpeak(idx)

		self.peaks = self.data
		#print(peaks)
		label = 1
		for peak in tqdm(range(len(self.data))):
			if self.labels[peak] == 0:
				# Compute distances between peaks
				distances = spatial.distance.cdist(np.array(self.peaks[peak][:]).reshape(1,-1), self.peaks, metric='euclidean')

				# Find all the coordinates of peaks where the distance is less than r/2
				coordinates = np.argwhere(distances < self.r/2)

				# Merge these peaks
				self.peaks[coordinates[:,1]] = self.peaks[coordinates[0,1]][:]

				# Label the peaks
				self.labels[coordinates[:,1]] = label
				label += 1


	def run(self):
		print(self.data)
		print(self.unedited_data)
		self.meanshift()
		print(np.unique(self.data - self.unedited_data))
		#print(self.data)
		#print(self.unedited_data)
		print(self.peaks)
		print(np.unique(self.labels))
		plotclusters3D(self.unedited_data, self.labels, self.peaks)

try1 = Segmentation(points, 2)
try1.run()

