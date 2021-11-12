---
layout: post
author: dinhhuyhoang
title: Bài 2 - Gradient Descent
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Lập công thức chung](#2-generalization)
- [3. Thực nghiệm với Python](#3-coding)
- [4. Learning rate](#4-learning-rate)
- [5. Đánh giá và kết luận](#5-evaluation)
- [6. Tham khảo](#6-references)

<a name="1-introduction"></a>

## 1. Giới thiệu

Ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html) chúng ta đã đi tìm nghiệm cho bài toán bằng các phương pháp sử dụng hình học, đại số tuyến tính kết hợp đạo hàm. Nhưng điểm yếu còn tồn tại đó là ta phải tính ma trận nghịch đảo (trong nhiều trường hợp không thể tìm trực tiếp) làm chậm về tốc độ tính toán, tràn bộ nhớ với những tập dữ liệu lớn. Thuật toán Gradient Descent có thể cải thiện hơn mà vẫn đạt độ hiệu quả khá cao.

Ý tưởng của thuật toán Gradient Descent chính là việc ứng dụng đạo hàm để tìm nghiệm tối ưu. Để dễ dàng giải thích, hãy xem đồ thị của phương trình $f(x) = x^4 - 5x^2 - x + 3$

<img src="/assets/images/bai2/anh1.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Đồ thị hàm số</p>

Để tìm giá trị nhỏ nhất cho hàm $f(x)$ trên. Ta thường tìm đạo hàm $f'(x)$ và tìm nghiệm $f'(x) = 0$. Và sau đó thế lại các nghiệm đã tìm được vào hàm $f(x)$ để lấy giá trị nhỏ nhất. Trong machine learning, quá trình thế này có thể gọi là so sánh các local optimum để lấy được global optimum. Ở đồ thị <b>Hình 1</b> ta có nghiệm $x_1$ là nghiệm local optimum và $x_2$ là nghiệm global optimum. Nhưng trong các bài toán machine learning rất khó để có thể tìm đạo hàm rồi đi tìm từng nghiệm, vì vậy ta sẽ cố gắng tìm các điểm local optimum nào đó và coi nó là nghiệm chấp nhận được cho bài toán.

Xét điểm $x_0$ trên đồ thị <b>Hình 1</b> là nghiệm khởi tạo ban đầu, điều ta cần chính là làm sao để $x_0$ tiến gần tới $x_2$. Đây chính là cách mà thuật toán Gradient Descent sẽ làm. Qua nhiều lần lặp với công thức $x_0 := x_0 - \alpha f'(x_0)$ trong đó hằng số dương $\alpha$ được gọi là learning rate cần được khởi tạo ban đầu. Ta thấy $f'(x_0) < 0 <=> \alpha f'(x_0) < 0 =>$ $x_0$ sẽ ngày càng tiến đến vị trí $x_2$ và làm cho $f(x_0)$ giảm dần. 

<a name="2-generalization"></a>

## 2. Lập công thức chung

Để ứng dụng thuật toán Gradient Descent, ở bài này mình sẽ trình cách sử dụng thuật toán này để giải quyết bài toán [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html). Vì vậy hàm dự đoán và hàm mất mát vẫn tương tự.

Bài toán dự đoán giá nhà dựa trên diện tích với $m$ mẫu dữ liệu, ta có:

$$X = \begin{bmatrix} 1 && x_1 \\ 1 && x_2 \\ ... && ... \\ 1 && x_m \end{bmatrix}, Y = \begin{bmatrix} y_1 \\ y_2 \\ ... \\ y_m \end{bmatrix}, W = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}, \hat{Y} = X.W = \begin{bmatrix} w_0*1 + w_1*x_1  \\ w_0*1 + w_1*x_2 \\ ... \\ w_0*1 + w_1*x_m \end{bmatrix} = \begin{bmatrix} \hat{y_1} \\ \hat{y_2} \\ ... \\ \hat{y_m} \end{bmatrix}$$

$$J = \frac{1}{2m}sum(\hat{Y}-Y)^2$$

$$\frac{dJ}{dW} = \frac{1}{m}X^T.(\hat{Y}-Y)$$

Các bước để xử lí bài toán:
- Bước 1: Khởi tạo nghiệm $W$, learning rate $\alpha$, số lần lặp $\textbf{iterations}$.
- Bước 2: Cập nhật $W := W - \alpha \frac{dJ}{dW}$.
- Bước 3: Dừng lại khi hết $\textbf{iterations}$ lần lặp.

Ở bước 3, có khá nhiều cách để dừng vòng lặp như:
- Giới hạn số lần lặp (như ở trên).
- Giới hạn giá trị hàm cost - J.
- Giới hạn hiệu giá trị của hàm cost tại 2 lần gần nhất.
- ...

Nhưng trong thực thế, cách giới hạn số vòng lặp là đơn giản nhất mà vẫn đạt được các giá trị tốt. Tham khảo thêm [các phương pháp dừng vòng lặp](https://machinelearningcoban.com/2017/01/16/gradientdescent2/#-stopping-criteria-dieu-kien-dung).

<a name="3-coding"></a>

## 3. Thực nghiệm với Python

```python

import numpy as np # ĐSTT
import matplotlib.pyplot as plt # Visualize

# Hàm này sử dụng để tìm nghiệm W và cost sau mỗi lần lặp
# vì vậy W và cost cuối cùng sẽ là nghiệm và giá trị cost cho bài toán
def process(W, X, Y, learning_rate, num_iterations):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.concatenate((ones, X),axis=1)
    cost = np.zeros((num_iterations,1))
    W_list = []
    for i in range(num_iterations):
        dist = np.dot(X,W) - Y
        cost[i] = 0.5*(1/m)*np.sum(dist*dist)
        W = W - learning_rate/m*(np.dot(X.T,dist))
        W_list.append(W)
    return W_list,cost

# Hàm này sử dụng để dự đoán khi có một giá trị x0
def predict(W,x0):
    w0,w1 = W[0],W[1]
    y0 = w0 + w1*x0
    return y0

# Dùng để visualize đường thẳng cần tìm
def draw_line(min_x,max_x,W,color='r'):
    min_y = predict(W,min_x)
    max_y = predict(W,max_x)
    plt.plot((min_x,max_x),(min_y,max_y),color)

def main():
    # Dữ liệu
    X = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T
    Y = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
    
    # Visualize dữ liệu
    plt.scatter(X, Y)
    min_x,max_x = 2,46 # Điểm để visualize
    
    # Khởi tạo nghiệm
    W = np.array([[1.],
                  [2.]])
    
    # leaning rate - alpha
    learning_rate = 0.0001
    
    # số lần lặp iterations
    num_iterations = 100
    
    # Visualize đường khởi tạo
    draw_line(min_x, max_x, W,color='black')
    
    # Tìm nghiệm
    W_list,cost = process(W, X, Y, learning_rate, num_iterations)
    
    # Visualize đường thẳng các nghiệm tìm được
    for W in W_list:
        draw_line(min_x,max_x,W,'blue')
        
    # Visualize đường kết quả
    draw_line(min_x,max_x,W_list[-1],'red')
    
    title = ['init line','update line']
    plt.tight_layout()
    plt.legend(title)
    plt.xlabel('Diện tích')
    plt.ylabel('Giá')
    plt.show()
```
<p style="display: flex">
<img src="/assets/images/bai2/anh2.png" class="smallpic"/> <img src="/assets/images/bai2/anh3.png" class="smallpic"/>
</p>
<p align="center"> <b>Hình 2</b>: Visualize các nghiệm sau mỗi lần cập nhật và đường thẳng tương ứng </p>

Ta thấy đường màu đen chính là nghiệm mà ta khởi tạo, các đường màu xanh chính là nghiệm sau mỗi cập nhật theo công thức Gradient Descent (ở bước 2). Đường màu đỏ chính là nghiệm cuối cùng của bài toán.

Ở đây ta thấy sau 100 lần lặp thì nghiệm cuối cùng đã tìm được, nghiệm cho bài toán khá tốt, đường tìm được khá giống với nghiệm tìm theo công thức ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html). Hãy xem hàm mất mất biến đổi như thế nào sau mỗi lần lặp

<img src="/assets/images/bai2/anh4.png" class="normalpic"/>

<p align="center"> <b>Hình 3</b>: Visualize cost function </p>

<a name="4-learning-rate"></a>

## 4. Learning rate

Có một điều khá qua trọng mà chưa được nhắc tới ở phần trên đó là việc chọn learning rate. Hãy xem sự ảnh hưởng của chọn learning rate

<p style="display: flex">
<img src="/assets/images/bai2/anh5.png" class="smallpic"/> <img src="/assets/images/bai2/anh6.png" class="smallpic"/>
</p>

<p style="display: flex">
<img src="/assets/images/bai2/anh8.png" class="smallpic"/> <img src="/assets/images/bai2/anh7.png" class="smallpic"/>
</p>

<p align="center"> <b>Hình 4</b>: So sánh chọn learning rate </p>

Ta thấy ở <b>Hình 4</b>, việc lựa chọn các learning rate khác nhau sẽ đem đến các kết quả khác nhau. Đây cũng là một hạn chế của thuật toán này khi phải tuning lại giá trị của learning rate nhiều lần. Nhưng nếu chọn learning rate quá nhỏ sẽ làm cho quá trình hội tụ (tức đạt tới giá trị nghiệm tối ưu) rất rất chậm, hoặc nếu quá to thì sẽ có thể không bao giờ đạt tới

<img src="/assets/images/bai2/anh9.png" class="normalpic"/>

<p align="center"> <b>Hình 5</b>: Hình trái - learning quá nhỏ, Hình phải - learning rate quá to </p>

<a name="5-evaluation"></a>

## 5. Đánh giá và kết luận

- Thuật toán Gradient Descent đã giải quyết được vấn đề về tính toán và bộ nhớ so với cách tính ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html)
- Tuy nhiên, chọn learning rate và điểm khởi tạo lại chính là điểm yếu của thuật toán này, vì vậy ta phải lựa chọn nhiều lần để có được kết quả mong muốn.
- Gradient Descent là tiền đề của rất nhiều thuật toán tối ưu nâng cao hơn như: Adam, RMSprop, SGD...
- Vì dựa trên đạo hàm nên việc đạo hàm chính xác một hàm số là điều tiên quyết, có 1 cách để debug xem mình đã đạo hàm đúng hay chưa dựa trên định nghĩa của đạo hàm. Tham khảo thêm [kiểm tra đạo hàm](https://machinelearningcoban.com/2017/01/12/gradientdescent/#ki%e1%bb%83m-tra-%c4%91%e1%ba%a1o-h%c3%a0m)

<a name="6-references"></a>

## 6. Tham khảo

[1] [Machine Learning cơ bản](https://machinelearningcoban.com/2017/01/12/gradientdescent/)

[2] [Dung Lai github page](https://dunglai.github.io/2017/12/21/gradient-descent/)