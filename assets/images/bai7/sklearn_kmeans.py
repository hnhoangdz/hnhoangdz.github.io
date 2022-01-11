from sklearn.cluster import KMeans
import matplotlib.pyplot as plt # Visualize
from sklearn.datasets import make_blobs # Make the dataset

# Init original centroids
true_centroids = [[1, 1], [-1, -1], [1, -1]] 

# dataset
X, true_labels = make_blobs(n_samples=750, centers=true_centroids, 
                            cluster_std=0.4, random_state=0)
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_

labels = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c = labels, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], color='black')