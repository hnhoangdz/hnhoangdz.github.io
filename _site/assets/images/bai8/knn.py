import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

print('Number of classes: %d' %len(np.unique(y)))
# >>> Number of classes: 3

print('Number of data points: %d' %X.shape[0])
# >>> Number of data points: 150

print('#Features of each point: %d'%X.shape[1])
# >>> Features of each point: 4

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=50)

print("Numbers of training set: %d" %X_train.shape[0])
print("Numbers of test set: %d" %X_test.shape[0])

# Khoảng cách o-clit giữa 2 điểm
def distance(p1,p2):
    return np.linalg.norm(p1-p2,2)

# Tính toàn bộ khoảng cách với điểm xét x0
def get_all_distance(x0, X, y):
    distances = []
    for i in range(len(X)):
        dist = distance(x0,X[i])
        # Lấy toàn bộ khoảng cách và nhãn tương ứng
        distances.append((dist,y[i]))
    return distances

# Lấy ra K điểm có khoảng cách ngắn nhất
def get_K_nearest_neighbors(K,X,y,x0):
    neighbors = []
    distances = get_all_distance(x0,X,y)
    # Sắp xếp tăng dần dựa trên khoảng cách kèm theo nhãn
    distances.sort(key=operator.itemgetter(0))
    for i in range(K):
        neighbors.append(distances[i][1])
    return neighbors

# Lấy ra nhãn có số vote cao nhất
def major_voting(neighbors,n_classes):
    counting = [0]*n_classes
    for i in range(n_classes):
        counting[neighbors[i]] += 1
    return np.argmax(counting)

# Hàm dự đoán
def predict(X,y,K,x0,n_classes):
    neighbors = get_K_nearest_neighbors(K,X,y,x0)
    y_pred = major_voting(neighbors,n_classes)
    return y_pred

# Tính phần trăm chính xác của dự đoán và test
def accuracy_score(y_preds,y_test):
    n = len(y_preds)
    count = 0
    for i in range(n):
        if y_preds[i] == y_test[i]:
            count += 1
    return count/n

def main():
    np.random.seed(7)
    K = 5
    n_classes = 3
    y_preds = []
    for xt in X_test:
        yt = predict(X_train, y_train, K, xt, n_classes)
        y_preds.append(yt)
    acc = accuracy_score(y_preds,y_test)
    print('Accuracy score: %f' %acc)

if __name__ == '__main__':
    main()