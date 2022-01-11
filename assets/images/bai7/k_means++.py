import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
  
# Init original centroids
true_centroids = [[1, 1], [-1, -1], [1, -1]] 

# Dataset
X, true_labels = make_blobs(n_samples=750, centers=true_centroids, 
                            cluster_std=0.4, random_state=0)
  
# Hàm để vẽ các bước chọn tâm cụm
def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker = '.',
                color = 'gray', label = 'data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color = 'black', label = 'previously centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color = 'red', label = 'next centroid')
    plt.title('Selected % d th centroid'%(centroids.shape[0]))
     
    plt.legend()
    plt.show()
          
# Tính khoảng cách 2 điểm
def distance(p1,p2):
    return np.linalg.norm(p1-p2,2)
  
# Thuật toán khởi tạo
def initialize(data, K):
    centroids = []
    # chọn ngẫu nhiên tâm cụm đầu tiên (bước 1)
    random_idx = np.random.randint(data.shape[0])
    centroids.append(data[random_idx])
    plot(data, np.array(centroids))
    for c_id in range(K - 1):

        dists = []
        
        # Bước 2
        # tìm khoảng cách của mỗi điểm dữ liệu với các tâm cụm đang có
        # chọn ra khoảng cách ngắn nhất
        
        for i in range(data.shape[0]):
            point = data[i, :]
            d = 999999
             
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dists.append(d)
             
        dists = np.array(dists)
        
        # Bước 3
        # lấy ra index có giá trị lớn nhất trong dists
        next_centroid = data[np.argmax(dists), :]
        
        # Bước 4 
        # Tâm cụm mới chính là giá trị của index trong dữ liệu ban đầu
        centroids.append(next_centroid)
        plot(data, np.array(centroids))
    return centroids
initialize(X,3)