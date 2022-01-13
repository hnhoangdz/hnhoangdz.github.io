import numpy as np # ĐSTT
import matplotlib.pyplot as plt # Visualize
from sklearn.datasets import make_blobs # Make the dataset

centers = [[1, 1], [-1, -1], [1, -1]] 
X, true_labels = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

# For visualize
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4.5))
ax1.plot(X[:, 0], X[:, 1], 'o')
ax1.set_title('Before')

labels = np.array([0]*len(true_labels))
visited = np.array([False]*len(true_labels))
labels_border = []
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4.5))
# ax1.plot(X[:, 0], X[:, 1], 'o')
# ax1.set_title('Before')

# Calculate distance by norm 2 of two points
def distance(p1,p2):
    return np.linalg.norm(p1-p2,2)

# Find nearest neighbors that have distance <= epsilon of point
def nearestNeighbors(data, p, eps):
    neighbors = []
    for i,point in enumerate(data):
        if distance(p, point) <= eps:
            neighbors.append(i)
    return neighbors

# Main algorithm - DBSCAN
def dbscan(data, eps, minPoints):
    c = 0 # label for each dense density
    for i in range(len(data)):
        p = data[i]
        if visited[i] == True: # this point was visited
            continue
        visited[i] = True # change point status to visited
        neighbors = nearestNeighbors(data, p, eps) # nearest neighbors of point
        if len(neighbors) < minPoints:
            labels[i] = -1 # this is noise point
            continue
        labels[i] = c # core point
        neighbors.remove(i) # remove current core point
        S = neighbors
        # spread from core point
        for j in S:
            if visited[j] == True:
                continue
            labels[j] = c # change point's label of nearest core point
            neighbors = nearestNeighbors(data, data[j], eps) # nearest neighbors of nearest core point
            if len(neighbors) >= minPoints:
                S += neighbors # spread from core point more
            else:
                labels_border.append(j) # this is boder point
            S.remove(j) # remove current core point
            visited[j] = True # change point status to visited
        c += 1 # new cluster initalizes 
        
        
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4.5))
ax1.plot(X[:, 0], X[:, 1], 'o')
ax1.set_title('Before')

# Main algorithm
eps=0.3
minPoints=10
dbscan(X, eps, minPoints)

# For visualize
n_noise_ = list(labels).count(-1)
core_samples_mask = np.ones_like(labels, dtype=bool)
core_samples_mask[np.array(labels_border)] = False
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

ax2.set_title('After')
plt.show()

print('Number of outlier: ', n_noise_)
print('Number of boder: ', len(labels_border))
print('Number of core: ', len(labels) - len(labels_border) - n_noise_)
