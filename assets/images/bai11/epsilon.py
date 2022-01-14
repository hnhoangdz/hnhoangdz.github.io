import numpy as np # ĐSTT
import matplotlib.pyplot as plt # Visualize
from sklearn.datasets import make_blobs # Make the dataset
from sklearn.neighbors import NearestNeighbors

centers = [[1, 1], [-1, -1], [1, -1]] 
X, true_labels = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
neighbors = 5
neigh = NearestNeighbors(n_neighbors=neighbors)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances[:,neighbors-1], axis=0)
from kneed import KneeLocator
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print('Epsilon = ', distances[knee.knee])
plt.plot(distances)