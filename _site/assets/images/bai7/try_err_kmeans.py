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
# labels = []

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

# K = 3
# centroids = init_centroids(K)
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
def update_centroids(centroids, X, labels):
    # print(centroids.shape)
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

def cost_function(centroids, X, labels):
    cost_f = 0
    for i in range(len(X)):
        cost_f += distance(centroids[labels[i]], X[i])
    return cost_f/X.shape[0]

final_centroids = []
final_labels = []
cost = 9999999

for i in range(500):
    K = 3
    centroids = init_centroids(K)
    labels = update_labels(X)
    centroids, _ = update_centroids(centroids, X, labels)
    labels, centroids = np.array(labels), np.array(centroids)
    cost_f = cost_function(centroids, X, labels)
    if cost_f < cost:
        cost = cost_f
        final_centroids = centroids
        final_labels = labels
print(true_labels)
print(final_centroids)
print(cost)
plt.scatter(X[:,0], X[:,1], c = final_labels, cmap='rainbow')
plt.scatter(final_centroids[:,0], final_centroids[:,1], color='black')


