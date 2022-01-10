import numpy as np # ĐSTT
import matplotlib.pyplot as plt # Visualize

# Hàm sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Sử dụng GD để tìm nghiệm
def process(W,X,Y,learning_rate,num_iterations):
    m = X.shape[0]
    ones = np.ones((m,1),dtype='float32')
    X = np.concatenate((ones,X),axis=1)
    cost = np.zeros((num_iterations,1))
    for i in range(num_iterations):
      y_pred = sigmoid(np.dot(X,W))
      cost[i] = -1/m*np.sum(np.multiply(Y,np.log(y_pred))+np.multiply((1-Y),np.log(1-y_pred)))
      W = W - learning_rate/m*np.dot(X.T,y_pred - Y)
    return cost,W

# Dự đoán 1 điểm dữ liệu mới
def predict(x1,x2,W,threshold):
    w0,w1,w2 = W[0][0],W[1][0],W[2][0]
    y_hat = sigmoid(w0 + w1*x1 + w2*x2)
    return 1 if y_hat >= threshold else 0

# Vẽ đường phân cách dựa trên ngưỡng
def draw_line(x21,x22,W,threshold):
    w0,w1,w2 = W[0][0],W[1][0],W[2][0]
    x11 = -(w0 + w1*x21 + np.log(1/threshold - 1))/w2
    x12 = -(w0 + w1*x22 + np.log(1/threshold - 1))/w2
    plt.plot((x21, x22),(x11,x12), 'g')
    plt.show()

def main():
    # Dữ liệu
    X = np.array([[10,5,6,7,8,9,4,5,8,4,8,7,4,5,7,4,5,6,7,8],
    [1,2,1.8,1,2,0.5,3,2.5,1,2.5,0.1,0.15,1,0.8,0.3,1,0.5,0.3,0.2,0.15]],dtype='float32').T
    
    Y = np.array([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]],dtype='float32').T

    X_cho_vay = X[Y[:,0]==1] # lấy các giá trị cho vay của X
    X_tu_choi = X[Y[:,0]==0] # lấy các giá trị từ chối của X
    
    # Visualize dữ liệu
    plt.scatter(X_cho_vay[:, 0], X_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
    plt.scatter(X_tu_choi[:, 0], X_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
    plt.legend(loc=1)
    plt.xlabel('mức lương (triệu)')
    plt.ylabel('kinh nghiệm (năm)')
    
    # Khởi tạo nghiêm ban đầu
    W = np.array([[0.,0.1,0.1]]).T
    
    # Khởi tạo learning rate
    learning_rate = 0.3
    
    # Khởi tạo số vòng lặp
    num_iterations = 1000
    
    # Nghiệm tìm được
    cost,W = process(W,X,Y,learning_rate,num_iterations)
    
    # Draw đường phân cách boudary dựa trên ngưỡng
    x21 = 4
    x22 = 10
    threshold = 0.5
    draw_line(x21,x22,W,threshold)
    
    print(W)
    
if __name__ == '__main__':
    main()