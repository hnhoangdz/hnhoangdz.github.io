import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier

# dataset
centers = [[1, 1], [-1, -1], [1, -1]] 
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.5], [1, 1]])

plt.scatter(X[:, 0], X[:, 1], c='b')
plt.scatter(X_new[:,0], X_new[:,1],marker="v",c='red')

# Init DBSCAN model and fit
db = DBSCAN(eps=0.18, min_samples=5).fit(X)
labels = db.labels_

# Init KNN model
knn = KNeighborsClassifier(n_neighbors = 5, weights='distance')
knn.fit(X, labels)

# Predict X_new
y_pred = knn.predict(X_new)
print('X_new label predict: ', y_pred)
print(knn.predict_proba(X_new))


