import numpy as np # ĐSTT
import random 
import matplotlib.pyplot as plt # Visualize
from sklearn.datasets import make_blobs # Make the dataset

# Init original centroids
true_centroids = [[1, 1], [-1, -1], [1, -1]] 

# dataset
X, true_labels = make_blobs(n_samples=750, centers=true_centroids, 
                            cluster_std=0.4, random_state=0)

# predict labels
labels = []

# # Visualize dữ liệu
# plt.plot(X[:,0],X[:,1],'o')
# plt.title('Dataset')

# Khởi tạo centroids
def init_centroids(K):
    centroids = []
    K = 3
    for i in range(K):
        x = random.sample(list(X), 1)
        centroids.append(x[0])
    centroids = np.array(centroids)
    return centroids

K = 3
centroids = init_centroids(K)
# plt.plot(centroids[:,0],centroids[:,1],'o')


# Tính khoảng cách 2 điểm
def distance(p1,p2):
    return np.linalg.norm(p1-p2,2)

# Cập nhật nhãn cho mỗi điểm dữ liệu
def update_labels(X):
    labels = []
    for i in range(len(X)):
        fake_distance = 999999
        label = -1
        for j in range(len(centroids)):
            d = distance(X[i],centroids[j])
            if d < fake_distance:
                fake_distance = d
                label = j
        labels.append(label)
    return labels

# Cập nhật centroids dựa trên nhãn trước đó tìm được
def update_centroids(centroids,X,labels):
    before_centroids = centroids.copy()
    for i in range(len(centroids)):
        count = 0
        x0 = 0
        y0 = 0
        for j in range(len(X)):
            if i == labels[j]:
                count += 1
                x0 += X[j][0]
                y0 += X[j][1]
        if count == 0:
            break
        x0 /= count
        y0 /= count
        centroids[i][0] = x0
        centroids[i][1] = y0
    return centroids, before_centroids

# Điều kiện dừng
def stop(centroids,new_centroids):
    return (set([tuple(a) for a in centroids]) == 
        set([tuple(a) for a in new_centroids]))

while True:
    labels = update_labels(X)
    centroids, before_centroids = update_centroids(centroids, X, labels)
    if stop(centroids, before_centroids):
        break
print(labels)
plt.scatter(X[:,0], X[:,1], c = labels, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], color='black')


