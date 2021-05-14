import scipy.io as sio
from scipy import spatial
from skimage import color
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import time

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
	print(peaks)
	peaks = color.lab2rgb(peaks)
	print(peaks)
	ax = fig.add_subplot(111, projection="3d")
	bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
	rgb_peaks = bgr_peaks[..., ::-1]
	rgb_peaks /= 255.0
	for idx, peak in enumerate(rgb_peaks):
		# color_value = np.random.uniform(0, 1, 3)
		#TODO: instead of random color, you can use peaks when you work on actual images
		color_value = peak
		cluster = data[np.where(labels == idx + 1)[0]].T
		ax.scatter(cluster[0], cluster[1], cluster[2], c=[color_value], s=.5)
	plt.show()

	labels = labels.reshape(360, 180, 3)
	image = np.zeros(labels.shape)
	image[labels]
	cv2.imshow('image', peaks)
	cv2.waitKey()

class imageSegmentation():
	def __init__(self, image_path, r, threshold, c, include_coordinates, image_name):
		points = cv2.imread(image_path)
		self.unedited_data = points.copy()
		points = color.rgb2lab(points)
		self.image_x = points.shape[0]
		self.image_y = points.shape[1]
		self.data, self.min_points, self.max_points = self.normalize(points, include_coordinates)
		print(self.data.shape)
		self.r = r
		self.threshold = threshold
		self.c = c
		self.labels = np.zeros(len(self.data))
		self.peaks = np.array([])
		self.time_taken = 0
		self.time_start = time.time()
		self.image_name = image_name


	def normalize(self, points, include_coordinates):
		unshaped_points = points.copy()
		points = np.array(points).reshape(-1, 3)
		points1 = points[:, 0]
		points2 = points[:, 1]
		points3 = points[:, 2]
		min_points = np.array([points1.min(), points2.min(), points3.min()])
		max_points = np.array([points1.max(), points2.max(), points3.max()])
		points[:, 0] = (points1 - points1.min()) / (points1.max() - points1.min())
		points[:, 1] = (points2 - points2.min()) / (points2.max() - points2.min())
		points[:, 2] = (points3 - points3.min()) / (points3.max() - points3.min())

		if include_coordinates:
			i_coords, j_coords = np.meshgrid(range(unshaped_points.shape[0]), range(unshaped_points.shape[1]), indexing='ij')
			v_1 = i_coords = i_coords.reshape(-1, 1)
			v_2 = j_coords = j_coords.reshape(-1, 1)
			i_coords[:] = (v_1 - v_1.min()) / (v_1.max() - v_1.min())
			j_coords[:] = (v_2 - v_2.min()) / (v_2.max() - v_2.min())
			coords = np.hstack((i_coords.reshape(-1, 1), j_coords.reshape(-1, 1)))
			return np.hstack((points, coords)), min_points, max_points
		else:
			return points, min_points, max_points

	def denormalize(self, points):
		points1 = points[:, 0]
		points2 = points[:, 1]
		points3 = points[:, 2]

		points[:, 0] = (points1 * (self.max_points[0] - self.min_points[0])) + self.min_points[0]
		points[:, 1] = (points2 * (self.max_points[1] - self.min_points[1])) + self.min_points[1]
		points[:, 2] = (points3 * (self.max_points[2] - self.min_points[2])) + self.min_points[2]
		return points

	def findpeak(self, idx):
		# centre of sphere
		centre = np.array(self.data[idx][:]).reshape(1, -1)
		check = 0
		while True:
			# Calculate the euclidean distance between the point at idx and the rest of the points.
			distances = spatial.distance.cdist(centre, self.data, metric='euclidean')

			# Find all the coordinates of points where the distance is less than r
			coordinates = np.argwhere(distances < self.r)

			# Array containing all the points which are in the radius
			points_in_radius = self.data[coordinates[:, 1]][:]

			# Check if we have reached the threshold and return the peak
			if spatial.distance.cdist(centre, np.average(points_in_radius, axis=0).reshape(1, -1), metric='euclidean') < self.threshold:
				return np.average(points_in_radius, axis=0)

			# Compute average of the points in radius and shift window
			centre = np.array(np.average(points_in_radius, axis=0)).reshape(1, -1)

	def meanshift(self):
		label = 1
		for idx in tqdm(range(len(self.data))):
			# Compute new peak.
			new_peak = self.findpeak(idx)

			# Check if we can merge this peak with a nearby one.
			if len(self.peaks) != 0:
				distances = spatial.distance.cdist(np.array(new_peak).reshape(1, -1), self.peaks, metric='euclidean')

				# Find all the coordinates of peaks where the distance is less than r/2
				coordinates = np.argwhere(distances < self.r/2)

				# Merge these peaks
				if len(coordinates) != 0:
					self.labels[idx] = coordinates[0][1] + 1

				else:
					# Label the peak
					self.labels[idx] = label
					label += 1

					# Store the peak
					self.peaks = np.vstack([self.peaks, new_peak.reshape(1, -1)])

			else:
				# Label the peak
				self.labels[idx] = label
				label += 1

				# Store the peak
				self.peaks = new_peak.reshape(1, -1)

	def findpeak_opt(self, idx):
		# Centre of sphere
		centre = np.array(self.data[idx][:]).reshape(1, -1)

		# Second speed up
		path_points = np.array([[0, idx]])

		while True:
			# Calculate the euclidean distance between the point at idx and the rest of the points.
			distances = spatial.distance.cdist(centre, self.data, metric='euclidean')

			# Find all the coordinates of points where the distance is less than r
			coordinates = np.argwhere(distances <= self.r)

			# Array containing all the points which are in the radius
			points_in_radius = self.data[coordinates[:, 1]][:]

			# Check if we have reached the threshold and return the peak
			if spatial.distance.cdist(centre, np.average(points_in_radius, axis=0).reshape(1, -1),
									  metric='euclidean') < self.threshold:
				return np.average(points_in_radius, axis=0), np.vstack([coordinates, path_points])

			# Second speed up store coordinates within r/c of the position
			coordinates = np.argwhere(distances < self.r/self.c)
			path_points = np.vstack([path_points, coordinates])

			# Compute average of the points in radius and shift window
			centre = np.array(np.average(points_in_radius, axis=0)).reshape(1, -1)

	def meanshift_opt(self):
		label = 1
		counter = 0
		for idx in tqdm(range(len(self.data))):
			# First speed up, only check labels which haven't already been labelled
			if self.labels[idx] == 0:
				counter += 1
				# Compute new peak and coordinates of points in sphere.
				new_peak, coordinates_in_sphere = self.findpeak_opt(idx)

				# Check if we can merge this peak with a nearby one.
				if len(self.peaks) != 0:
					distances = spatial.distance.cdist(np.array(new_peak).reshape(1, -1), self.peaks, metric='euclidean')

					# Find all the coordinates of peaks where the distance is less than r/2
					coordinates = np.argwhere(distances < self.r / 2)

					# Merge these peaks
					if len(coordinates) != 0:
						self.labels[coordinates_in_sphere[:, 1]] = coordinates[0][1] + 1
						self.labels[idx] = coordinates[0][1] + 1

					else:
						# Label the peak
						self.labels[coordinates_in_sphere[:, 1]] = label
						self.labels[idx] = label
						label += 1

						# Store the peak
						self.peaks = np.vstack([self.peaks, new_peak.reshape(1, -1)])

				else:
					# Label the peak
					self.labels[coordinates_in_sphere[:, 1]] = label
					self.labels[idx] = label
					label += 1

					# Store the peak
					self.peaks = new_peak.reshape(1, -1)
		self.time_taken = time.time() - self.time_start

	def run(self):
		self.meanshift()
		self.plot()

	def run_opt(self):
		self.meanshift_opt()
		self.plot()

	def plot(self):
		segmented = self.peaks[self.labels.astype(int) - 1][..., :3]
		segmented = self.denormalize(segmented)
		segmented = color.lab2rgb(segmented)
		segmented = segmented.reshape(self.image_x, self.image_y, 3)
		# self.unedited_data = color.rgb2lab(self.unedited_data)
		# self.unedited_data = color.lab2rgb(self.unedited_data)
		# image = np.hstack((self.unedited_data, segmented))
		image = segmented
		image = image * 255
		# cv2.imshow("Main", image)
		cv2.imwrite(self.image_name + ".jpg", image)
		cv2.waitKey()



# image_path = r'C:\Users\Gebruiker\PycharmProjects\CV\Assignment_1_Segmentation\image3.jpg'
# # points = sio.loadmat(r"C:\Users\Gebruiker\PycharmProjects\CV\Assignment_1_Segmentation\data\pts.mat")["data"].transpose()
#
#
# # points = cv2.imread(image_path)
# # points = color.rgb2lab(points)
#
# try1 = imageSegmentation(image_path, 0.15, 0.001, 4, True, "trying1")
# try1.run_opt()
# print(try1.peaks.shape)
# try2 = Segmentation(points, 2, 0.01, 4)
# try2.run()


