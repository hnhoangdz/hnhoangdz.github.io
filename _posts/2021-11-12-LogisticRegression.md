---
layout: post
author: dinhhuyhoang
title: Bài 3 - Logistic Regression
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Hàm Sigmoid](#2-sigmoid-function)
- [3. Hàm dự đoán](#3-prediction)
- [4. Hàm mất mát](#4-loss)
- [5. Lập công thức chung](#5-generalization)
    - [5.1. Một điểm dữ liệu](#51-generalization1)
    - [5.2. Toàn tập dữ liệu với ma trận và vector hóa](#52-generalization2)
- [6. Thực nghiệm với Python](#6-coding)
- [7. Đánh giá và kết luận](#7-evaluation)
- [8. Tham khảo](#8-references)

<a name="1-introduction"></a>

## 1. Giới thiệu

Ở [bài 1 - LinearRegression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html) và [bài 2 - Gradient Descent](https://hnhoangdz.github.io/2021/11/10/Gradient-Descent.html) ta đã tìm hiểu về bài toán dự đoán (prediction) trong lớp bài toán Regression, ở bài này ta cùng tìm hiểu về bài toán phân loại (classification). Sự khác biệt cơ bản của lớp bài toán dự đoán và phân loại chính là đầu ra (output), vì giá trị dự đoán đầu ra sẽ không bị giới hạn và liên tục còn giá trị phân loại sẽ bị chặn và rời rạc. Ứng dụng của phân loại là rất nhiều trong đời sống, một số ví dụ đơn giản nhất như: phân loại nợ xấu, phân loại email spam, phân loại bệnh...

Cụ thể trong bài này ta cùng tìm hiểu về thuật toán Logistic Regression, dùng phần lớn phân loại nhị nhân (binary classification). Hiểu đơn giản phân loại nhị phân là đầu ra sẽ thuộc nhãn 0 hoặc 1. Ví dụ: nợ xấu - 1, không nợ xấu - 0; mắc ung thư - 1, không mắc ung thư - 0. Mục tiêu của phân loại nhị phân giúp dự báo xác suất xảy ra của một sự kiện, với giá trị xác suất tương ứng với nhãn 1 và nhãn 0 có tổng bằng 1. Xem hình dưới đây

<img src="/assets/images/bai3/anh1.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Bài toán nhị phân (<b>Source: </b><a href="https://machinelearningcoban.com/2017/01/21/perceptron/">machine learning cơ bản</a>)</p>

Ở hình trên, ta có 2 nhãn với: nhãn 1 đại diện cho các hình vuông màu xanh dương, nhãn 0 đại diện cho các hình tròn màu đỏ. Điều ta cần làm đó là dự đoán hình tam giác màu xám sẽ thuộc nhãn nào. Nhìn đơn giản ta có thể thấy, hình tam giác gần với các hình tròn màu đỏ hơn. Vì vậy ta mong xác suất mà thuộc nhãn 0 sẽ lớn hơn xác suất thuộc nhãn 1, và tổng xác suất của chúng bằng 1.

<a name="2-sigmoid-function"></a>

## 2. Hàm Sigmoid

Như đã đề cập ở trên, ta cần đầu ra là xác suất tương ứng cho mỗi nhãn. Do đó, hàm Sigmoid được ra đời để nhằm giới hạn giá trị đầu ra của bài toán Linear Regression. 

Hàm Sigmoid: $f(x) = \frac{1}{1+e^{-x}}$, ta thấy đồ thị hàm Sigmoid bên dưới có giá trị giới hạn [0,1]. Thật vậy:

$$\lim_{x \rightarrow +\infty} \sigma(x) = \lim_{x \rightarrow +\infty} \frac{1}{1+e^{-x}}=1$$

$$\lim_{x \rightarrow -\infty} \sigma(x) = \lim_{x \rightarrow -\infty} \frac{1}{1+e^{-x}}=0$$

<img src="/assets/images/bai3/anh2.png" class="normalpic"/>

<p align="center"> <b>Hình 2</b>: Đồ thị hàm Sigmoid</p>

Hãy thử đạo hàm hàm hàm Sigmoid, ta được:

$$f'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} - \frac{1}{(1+e^{-x})^2} = f(x) - f(x)^2 = f(x)(1-f(x))$$

Đạo hàm của hàm Sigmoid rất đẹp, giúp cho việc tính toán dễ dàng. 

<a name="3-prediction"></a>

## 3. Hàm dự đoán

Bắt đầu với bài toán phân loại khách hàng để quyết định cho vay hoặc từ chối dựa trên mức lương và số năm kinh nghiệm. Xem hình sau

<img src="/assets/images/bai3/anh3.png" class="normalpic"/>

<p align="center"> <b>Hình 3</b>: Visualize dữ liệu (<b>Source: </b><a href="https://nttuan8.com/bai-2-logistic-regression/">deep learning cơ bản</a>)</p>

Ta thấy với dữ liệu như trên ta cần tìm một đường thẳng để phân chia thành 2 vùng khác nhau bao gồm: vùng cho vay - nhãn 1, vùng từ chối - nhãn 0.

Giả sử có một người làm việc với mức lương là $x_1$, số năm kinh nghiệm là $x_2$ thì ta sẽ có giá trị dự đoán là:

$$\hat{y} = sig(w_0 + w_1x_1 + w_2x_2) = \frac{1}{1+e^{-({w_0 + w_1x_1 + w_2x_2})}}$$

trong đó $w_0, w_1, w_2$ là các giá trị ta cần tìm và cũng là hệ số đường thẳng phân chia (boundary), sig là kí hiệu cho hàm Sigmoid và $\hat{y}$ chính là giá trị dự đoán.

Lúc này giá trị $\hat{y}$ sẽ nằm trong khoảng [0,1]. Vì vậy thường ta sẽ đặt ngưỡng là 0.5, nếu $\hat{y} >= 0.5$ ta có thể coi là nhãn 1 (vùng cho vay) và ngược lại. Tất nhiên, nếu khó tính hơn thì có thể đặt ngưỡng cao hơn 0.5 để lấy được những người có thu nhập và kinh nghiệm chất lượng hơn.

<a name="4-loss"></a>

## 4. Hàm mất mát

Hàm mất mát dược sử dụng để đo độ sai số của mô hình, tức làm cho sai số là nhỏ nhất có thể. Tức nếu trong dữ liệu mẫu (training set) giá trị của một sample thứ $i$ là 1 thì giá trị dự đoán $\hat{y}$ cũng cần gần 1 và tương tự với 0.

Hàm mất mát ở bài toán này sử dụng là Binary Cross Entropy, để cho dễ dàng giải thích ở đây hàm loss sẽ biểu diễn cho một điểm dữ liệu:

$$L = -(y\log(\hat{y}) + (1-y)\log({1-\hat{y}}))$$

trong đó $y$ là giá trị thực 0 hoặc 1, $\hat{y}$ là giá trị dự báo nằm trong khoảng [0,1].

Hàm loss trên, có thể chia thành 2 trường hợp: 

-- Nếu $y = 1 => L = -\log{(\hat{y})}$

<img src="/assets/images/bai3/anh4.png" class="normalpic"/>

<p align="center"> <b>Hình 4</b>: Visualize Loss với $y = 1$ (<b>Source: </b><a href="https://nttuan8.com/bai-2-logistic-regression/">deep learning cơ bản</a>)</p>

Ta thấy:

- Khi giá trị của $\hat{y}$ và $y$ gần bằng nhau (tức $\hat{y}$ gần bằng 1) thì giá trị Loss sẽ giảm dần và ngược lại.

-- Nếu $y = 0 => L = -\log{(1 - \hat{y})}$

<img src="/assets/images/bai3/anh5.png" class="normalpic"/>

<p align="center"> <b>Hình 5</b>: Visualize Loss $y = 0$ (<b>Source: </b><a href="https://nttuan8.com/bai-2-logistic-regression/">deep learning cơ bản</a>)</p>

Ta thấy:

- Khi giá trị của $\hat{y}$ và $y$ gần bằng nhau (tức $\hat{y}$ gần bằng 0) thì giá trị Loss sẽ giảm dần và ngược lại.

Vì vậy, tính sai số của hàm Loss luôn được đảm bảo giữa giá trị dự đoán và giá trị thực. Hơn nữa, đồ thị luôn có cực trị tại $\hat{y} = y$.

<a name="5-generalization"></a>

## 5. Lập công thức chung

Với bài toán dự đoán cho vay hoặc từ chối nêu trên, đầu vào sẽ gồm 2 giá trị: lương và số năm kinh nghiệm. Đầu ra sẽ là xác suất tương ứng: nếu lớn hơn 0.5 thì sẽ là cho vay, nhỏ hơn 0.5 sẽ là từ chối.

<a name="51-generalization1"></a>

### 5.1. Một điểm dữ liệu

Với một điểm dữ liệu thứ $i$ trong tập dữ liệu gồm $m$ mẫu, ta có:

$$\hat{y_i} = sig(w_0 + w_1x_1 + w_2x_2)$$

$$L = -(y_i \log(\hat{y_i}) + (1 - y_i)\log(1-\hat{y_i}))$$

trong đó $x_1$ và $x_2$ là giá trị lương và số năm kinh nghiệm; $w_0, w_1$ và $w_2$ là biến cần tìm; $\hat{y_i}$ là giá trị dự đoán; sig là kí hiệu hàm Sigmoid; $y_i$ là giá trị thực.

Để tìm nghiệm cho bài toán này, ta sẽ sử dụng thuật toán Gradient Descent để tìm nghiệm (vì không thể giải và tìm nghiệm trực tiếp như bài Linear Regression).

Đặt $z = w_0 + w_1x_1 + w_2x_2 => \hat{y_i} = sig(z)$. Ta có:

$$\frac{dL}{dw_0} = \frac{dL}{dz}.\frac{dz}{dw_0} (1)$$

$$\frac{dL}{dw_1} = \frac{dL}{dz}.\frac{dz}{dw_1} (2)$$

$$\frac{dL}{dw_2} = \frac{dL}{dz}.\frac{dz}{dw_2} (3)$$

Lại có: 

$$\frac{dL}{dz} = \frac{dL}{d\hat{y_i}}.\frac{d\hat{y_i}}{dz} = \hat{y_i} - y_i (4)$$

$$\frac{dz}{dw_0} = 1 (5)$$

$$\frac{dz}{dw_1} = x_1 (6)$$

$$\frac{dz}{dw_2} = x_2 (7)$$

Từ (1),(4),(5) suy ra

$$\frac{dL}{dw_0} = \hat{y_i} - y_i$$

Từ (2),(4),(6) suy ra

$$\frac{dL}{dw_1} = x_1(\hat{y_i} - y_i)$$

Từ (3),(4),(7) suy ra

$$\frac{dL}{dw_2} = x_2(\hat{y_i} - y_i)$$

Lúc này ta tiến hành cập nhật các nghiệm với learning rate $\alpha$:

$$w_0 := w_0 - \alpha \frac{dL}{dw_0}$$

$$w_1 := w_1 - \alpha \frac{dL}{dw_1}$$

$$w_2 := w_2 - \alpha \frac{dL}{dw_2}$$

<a name="52-generalization2"></a>

### 5.2. Toàn tập dữ liệu với ma trận và vector hóa

Cho m điểm dữ liệu và learning rate $\alpha$, ta có:

$$X = \begin{bmatrix} 1&&x_1^{(1)}&&x_2^{(1)} \\1&&x_1^{(2)}&&x_2^{(2)} \\ ...&&...&&... \\ 1&&x_1^{(m)}&&x_2^{(m)}
\end{bmatrix}, Y = \begin{bmatrix}y_1\\ y_2 \\ ...\\ y_n\end{bmatrix},
W = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \end{bmatrix}$$

$$\hat{Y} = sig(X.W) = sig(\begin{bmatrix}  w_0 + w_1x_1^{(1)} + w_2x_2^{(1)}\\ w_0 + w_1x_1^{(2)} + w_2x_2^{(2)} \\ ... \\ w_0 + w_1x_1^{(m)} + w_2x_2^{(m)}\end{bmatrix}) = \begin{bmatrix} \hat{y_1} \\ \hat{y_2} \\ ... \\ \hat{y_m}\end{bmatrix}$$

$$J = -\frac{1}{m}\sum_{i=1}^{m}[Y\log(\hat{Y}) + (1 - Y)\log{(1 - \hat{Y}})]$$

$$\frac{dJ}{dW} = \frac{1}{m}X^T(\hat{Y}-Y)$$

$$W := W - \alpha \frac{dL}{dW}$$

<a name="6-coding"></a>

## 6. Thực nghiệm với Python

```python
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
def draw_line(x1,x2,W,threshold):
    w0,w1,w2 = W[0][0],W[1][0],W[2][0]
    y1 = -(w0 + w1*x1 + np.log(1/threshold - 1))/w2
    y2 = -(w0 + w1*x2 + np.log(1/threshold - 1))/w2
    plt.plot((x1, x2),(y1,y2), 'g')
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
    x1 = 4
    x2 = 10
    threshold = 0.5
    draw_line(x1,x2,W,threshold)
    
if __name__ == '__main__':
    main()
```

<img src="/assets/images/bai3/anh6.png" class="normalpic"/>

<p align="center"> <b>Hình 6</b>: Kết quả cho bài toán</p>

Chú ý: Ở hàm <b>draw_line</b>, ta cần dựa vào ngưỡng $\textbf{threshold}$ để vẽ được đường phân cách. Ở ví dụ đơn trên, ngưỡng $\textbf{threshold}$ được chọn là 0.5, để vẽ được đường thẳng đó ta cần 2 điểm dựa trên tọa độ.

Ta thấy với $\hat{y}$ dự đoán được, nếu $\hat{y} >= \textbf{threshold}$ tức nhãn một và cho vay tiền. Ta được:

$$\hat{y} >= \textbf{threshold}$$  

$$<=> \frac{1}{1+e^{-(w_0 + w_1x_1 + w_2x_2)}} >= \textbf{threshold} $$

$$<=> e^{-(w_0 + w_1x_1 + w_2x_2)} <= \frac{1}{\textbf{threshold}} - 1$$

$$<=> -(w_0 + w_1x_1 + w_2x_2) <= \ln(\frac{1}{\textbf{threshold}} - 1)$$

$$<=> x_2 >= \frac{-(w_0 + w_1 + \ln(\frac{1}{\textbf{threshold}} - 1))}{w_2} $$

Vậy để lấy điểm phân chia thì $ x_2 = \frac{-(w_0 + w_1 + \ln(\frac{1}{\textbf{threshold}} - 1))}{w_2} $. Lưu ý, $x_2$ ở đây chính là $y_1$ và $y_2$ ở hàm <b>draw_line</b> cần tìm. Chi tiết hơn xem [tại đây](https://nttuan8.com/bai-2-logistic-regression/#Quan_he_giua_phan_tram_va_duong_thang).

<a name="7-evaluation"></a>

## 7. Đánh giá và kết luận

- Logistic Regression có thể coi là một mạng neural network không có hidden layer, chỉ có input layer và output layer.
- Trong thực tế, nếu dữ liệu có các dạng phân bố theo nhãn phức tạp thì Logistic Regression hoạt động không hiệu quả vì hàm dự đoán rất khó có thể tìm ra đường thẳng, mặt phẳng hoặc siêu phẳng phù hợp. Do vậy việc neural network ra đời sẽ giúp cải tiến điểm yếu này.
- Ở loss function của Logistic Regression, chúng ta sử dụng hàm Binary Cross Entropy vì là hàm convex tức local optimum và global optimum là một.
- Ngoài ra tiền đề của thuật toán này và mạng neural network chính là thuật toán PLA, chi tiết hơn [tại đây](https://machinelearningcoban.com/2017/01/21/perceptron/)

<a name="8-references"></a>

## 8. Tham khảo

[1] [Machine Learning cơ bản](https://machinelearningcoban.com/2017/01/27/logisticregression/)

[2] [Deep Learning cơ bản](https://nttuan8.com/bai-2-logistic-regression/)


