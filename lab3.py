import numpy as np 
import matplotlib.pyplot as plt 
import random 

data = np.loadtxt('Lab3.txt')
np.random.shuffle(data)

plt.scatter(x = data[:,0], y = data[:,1], s = 2)
plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
	
			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.00) > self.tol:
					optimized = False

			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroids]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification 

clf = K_Means()
clf.fit(data)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
		marker="o", color="k", s=15, linewidths=5)
for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=5, linewidths=3)

plt.show()