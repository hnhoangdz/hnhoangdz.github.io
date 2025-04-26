from sklearn.linear_model import LogisticRegression
import numpy as np # ĐSTT
import matplotlib.pyplot as plt # Visualize

# Dữ liệu
X = np.array([[10,5,6,7,8,9,4,5,8,4,8,7,4,5,7,4,5,6,7,8],
[1,2,1.8,1,2,0.5,3,2.5,1,2.5,0.1,0.15,1,0.8,0.3,1,0.5,0.3,0.2,0.15]],dtype='float32').T

Y = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],dtype='float32')

X_cho_vay = X[Y == 1] # lấy các giá trị cho vay của X
X_tu_choi = X[Y == 0] # lấy các giá trị từ chối của X
plt.scatter(X_cho_vay[:, 0], X_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(X_tu_choi[:, 0], X_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')

log_reg = LogisticRegression()
log_reg.fit(X, Y)

print('b = {}, W = {}'.format(log_reg.intercept_, log_reg.coef_))

# Vẽ đường phân cách dựa trên ngưỡng
def draw_line(x21,x22,threshold):
    w0,w1,w2 = log_reg.intercept_[0], log_reg.coef_[0][0],log_reg.coef_[0][1]
    x11 = -(w0 + w1*x21 + np.log(1/threshold - 1))/w2
    x12 = -(w0 + w1*x22 + np.log(1/threshold - 1))/w2
    plt.plot((x21, x22),(x11,x12), 'g')
    plt.show()

draw_line(4,10,0.5)