import numpy as np # đstt
import matplotlib.pyplot as plt # visualize

# Dữ liệu
X = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T
Y = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T

# Visualize dữ liệu
plt.scatter(X, Y)

# thêm vector cột 1 vào dữ liệu input
m = X.shape[0]
ones = np.ones((m,1), dtype=np.int8)
X_square = np.square(X)
X = np.concatenate((ones,X), axis=1)
X = np.concatenate((X,X_square),axis=1)

# tìm nghiệm W cho bài toán
W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
w0,w1,w2 = W[0][0],W[1][0],W[2][0]

# lấy 10000 điểm để vẽ 
# để vẽ đường thẳng cần tìm
x0 = np.linspace(2,25,10000)
y0 = w0 + w1*x0 + w2*(x0**2)

# visualize đường thẳng cần tìm
plt.plot(x0,y0,'r')
plt.xlabel('Diện tích')
plt.ylabel('Giá nhà')
plt.show()

print(W)