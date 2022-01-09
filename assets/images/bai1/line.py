import numpy as np # đstt
import matplotlib.pyplot as plt # visualize

# Dữ liệu
X = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
Y = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

# Visualize dữ liệu
plt.scatter(X, Y)

# thêm vector cột 1 vào dữ liệu input
m = X.shape[0]
ones = np.ones((m,1), dtype=np.int8)
X = np.concatenate((ones,X), axis=1)

# tìm nghiệm W cho bài toán
W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
w0,w1 = W[0][0],W[1][0]

# lấy điểm đầu và điểm cuối 
# để vẽ đường thẳng cần tìm
x0 = np.linspace(2,46,2)
y0 = w0 + w1*x0

# visualize đường thẳng cần tìm
plt.plot(x0,y0,'r')
plt.xlabel('Diện tích')
plt.ylabel('Giá nhà')
plt.show()
print(X.shape)
print(X.T.dot(X).shape)