---
layout: post
author: dinhhuyhoang
title: 2. Gradient Descent
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Lập công thức chung](#2-generalization)
- [3. Thực nghiệm với Python](#3-coding)
- [4. Một số vấn đề lưu ý](#4-problems)
    - [4.1. Learning rate](#41-learning-rate)
    - [4.2. Cost function](#42-cost-function)
    - [4.3. Feature scaling](#43-feature-scaling)
- [5. Đánh giá và kết luận](#5-evaluation)
- [6. Tham khảo](#6-references)

<a name="1-introduction"></a>

## 1. Giới thiệu

Ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html) chúng ta đã đi tìm nghiệm cho bài toán bằng các phương pháp sử dụng hình học, đại số tuyến tính kết hợp đạo hàm. Nhưng điểm yếu còn tồn tại đó là ta phải tính ma trận nghịch đảo (trong nhiều trường hợp không thể tìm trực tiếp) làm chậm về tốc độ tính toán, tràn bộ nhớ với những tập dữ liệu lớn đặc biệt là khi số lượng features của dữ liệu rất lớn. Thuật toán Gradient Descent có thể cải thiện hơn mà vẫn đạt độ hiệu quả khá cao, hơn nữa Gradient Descent là cơ sở của rất nhiều thuật toán tối ưu trong Machine Learning/Deep Learning.

Ý tưởng của thuật toán Gradient Descent chính là việc ứng dụng đạo hàm để tìm nghiệm tối ưu. Để dễ dàng giải thích, hãy xem đồ thị của phương trình $f(x) = x^4 - 5x^2 - x + 3$

<img src="/assets/images/bai2/anh1.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Đồ thị hàm số</p>

Để tìm giá trị nhỏ nhất cho hàm $f(x)$ trên. Ta thường tìm đạo hàm $f'(x)$ và tìm nghiệm $f'(x) = 0$. Và sau đó thế lại các nghiệm đã tìm được vào hàm $f(x)$ để lấy giá trị nhỏ nhất. Trong machine learning, quá trình thế này có thể gọi là so sánh các local optimum để lấy được global optimum. Ở đồ thị <b>Hình 1</b> ta có nghiệm $x_1$ là nghiệm local optimum và $x_2$ là nghiệm global optimum. Nhưng trong các bài toán machine learning rất khó để có thể tìm đạo hàm rồi đi tìm từng nghiệm, vì vậy ta sẽ cố gắng tìm các điểm local optimum nào đó và coi nó là nghiệm chấp nhận được cho bài toán.

Xét điểm $x_0$ trên đồ thị <b>Hình 1</b> là nghiệm khởi tạo ban đầu, điều ta cần chính là làm sao để $x_0$ tiến gần tới $x_2$. Đây chính là cách mà thuật toán Gradient Descent sẽ làm. Qua nhiều lần lặp với công thức $x_0 := x_0 - \alpha f'(x_0)$ trong đó hằng số dương $\alpha$ được gọi là learning rate cần được khởi tạo ban đầu. Ta thấy $f'(x_0) < 0 <=> \alpha f'(x_0) < 0 =>$ $x_0$ sẽ ngày càng tiến đến vị trí $x_2$ và làm cho $f(x_0)$ giảm dần. **Tức ta cần di chuyển nghiệm $x_0$ ngược dấu với đạo hàm đang xét.**

<a name="2-generalization"></a>

## 2. Lập công thức chung

Để ứng dụng thuật toán Gradient Descent, ở bài này mình sẽ trình cách sử dụng thuật toán Gradient Descent để giải quyết bài toán hồi quy đã được xử lí bằng cách giải và tìm nghiệm trực tiếp ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html). Vì vậy hàm dự đoán và hàm mất mát vẫn tương tự. Như đã trình bày bên trên, ta sẽ sử dụng phương pháp cập nhật nghiệm ban đầu khởi tạo nhiều lần nhằm giảm giá trị hàm cost nhất có thể.

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

Nhưng trong thực thế, các giá trị hàm cost sẽ khá khó để biết nó thay đổi như thế nào khi train model, cách giới hạn số vòng lặp và đưa ra một số rules để thuật toán dừng (callbacks) là đơn giản nhất mà vẫn đạt được các giá trị tốt.

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

Ở đây ta thấy sau 100 lần lặp thì nghiệm cuối cùng đã tìm được, nghiệm cho bài toán khá tốt, đường tìm được khá giống với nghiệm tìm theo công thức ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html). Hãy xem hàm mất mát biến đổi như thế nào sau mỗi lần lặp

<img src="/assets/images/bai2/anh4.png" class="normalpic"/>

<p align="center"> <b>Hình 3</b>: Visualize cost function </p>

Trục ngang là giá trị số lần lặp của thuật toán, trục dọc là giá trị hàm cost sau mỗi vòng lặp. Nhận thấy rằng, tại vòng lặp thứ 40 đổ đi gần như hàm cost vẫn giữ nguyên giá trị rất gần 0, tức ta đã đạt được mong muốn rằng giá trị hàm cost càng nhỏ càng tốt (gần 0). Tuy nhiên, máy tính vẫn thực thi 60 vòng lặp mà các giá trị của hàm cost gần như không thay đổi vì vậy ta sẽ các chiến thuật để dừng vòng lặp là callbacks hoặc tính toán 2 giá trị cost gần nhất, về vấn đề này mình sẽ trình bày ở bài khác.

<a name="4-problems"></a>

## 4. Một số vấn đề lưu ý

<a name="41-learning-rate"></a>

### 4.1. Learning rate

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

<a name="42-cost-function"></a>

### 4.2. Cost function

Trong bài 1 và bài 2 này ta đã làm quen với Cost function là trung bình bình phương lỗi của toàn bộ dữ liệu, tên gọi tiếng Anh là: Mean Squared Error. Vậy tại sao hàm Cost function này lại được sử dụng mà không phải hàm số nào khác cũng có thể đo lường sai số giữa giá trị thực và giá trị dự đoán, cùng xem ví dụ sau:

<img src="/assets/images/bai2/anh10.jpg" class="normalpic"/>

<p align="center"> <b>Hình 6</b>: Gradient Descent pitfalls (Nguồn: Hands-on machine learning) </p>

Ở **Hình 6**, giả sử ta khởi tạo giá trị nghiệm ban đầu phía bên trái của điểm Local minimum thì hàm cost sẽ hội tụ tại đúng điểm Local minimum và rất khó có thể tới vươn tới điểm global minimum (gần như không thể). Mặt khác, nếu nghiệm ban đầu ta khởi tạo nằm trên đoạn Plateau thì việc cập nhật GD sẽ rất lâu tới điểm Global minimum, thậm chí nếu vòng lặp không đủ lớn nó sẽ dừng trước khi đạt tới điểm hội tụ Global minimum.

Đây cũng là lí do mà hàm Cost function cho bài toán Linear Regression là hàm Mean Squared Error. Hàm này đảm bảo yếu tố Convex khi tìm nghiệm tối ưu, hàm Convex sẽ giúp nghiệm bài toán luôn đảm bảo là Global minimum. Ví dụ hình ảnh:

<img src="/assets/images/bai2/anh12.png" class="normalpic"/>

<p align="center"> <b>Hình 7</b>: Convex vs Non-convex</p>

Ở **Hình 7**, phía bên trái là hàm Convex, bên phải không phải hàm Convex. Hiểu đơn giản rằng khi lấy 2 điểm bất kì trên đồ thị và nối chúng lại sẽ được một đường thẳng, nếu đường thẳng này không cắt bất kì điểm nào trên đồ thị nữa sẽ là hàm Convex. Việc tìm ra hàm Convext sẽ giúp ta dễ dàng giải quyết bài toán tối ưu rất nhiều, nghiệm sẽ luôn đảm bảo rằng là Gloabal minimum nếu số vòng lặp phù hợp và learning rate không quá lớn.

<a name="43-feature-scaling"></a>

### 4.3. Feature scaling

Trong thực tế, với mỗi loại dữ liệu thì sẽ có kharong giá trị khác nhau, ví dụ: diện tích nhà có thể có giá trị tới hàng trăm, hàng nghìn (mét vuông) tuy nhiên số phòng ngủ thì chỉ có thể là hàng đơn vị (thông thường). Vì vậy khi tiền xử lí dữ liệu, tất cả các thuật toán Machine Learning cần các kiểu dữ liệu về cùng một khoảng nhất định, điều này sẽ giúp máy tính giảm chi phí tính toán rất nhiều và quan trọng nhất là sẽ ảnh hưởng tới performance của thuật toán. Có 2 phương pháp scaling phổ biến nhất giúp đưa các loại dữ liệu về cùng 1 range [0,1] hoặc [-1,1]:

- Min-max scaling [0,1]: 

$$\begin{equation}
x’=\frac{x-min(x)}{max(x)-min(x)}
\end{equation}$$

trong đó $x$ là vector dữ liệu ban đầu

- Standard scaling [-1,1]: 

$$\begin{equation}
x’=\frac{x-mean(x)}{s_i}
\end{equation}$$

trong đó $s_i$ có thể là std (độ lệch chuẩn) hoặc range của $x$ (max - min)

<a name="5-evaluation"></a>

## 5. Đánh giá và kết luận

- Thuật toán Gradient Descent đã giải quyết được vấn đề về tính toán và bộ nhớ so với cách tính ở [bài 1 - Linear Regression](https://hnhoangdz.github.io/2021/11/06/LinearRegression.html)

- Tuy nhiên, chọn learning rate và điểm khởi tạo lại chính là điểm yếu của thuật toán này, vì vậy ta phải lựa chọn nhiều lần để có được kết quả mong muốn.

- Gradient Descent là tiền đề của rất nhiều thuật toán tối ưu nâng cao hơn như: Adam, RMSprop, SGD...

- Vì dựa trên đạo hàm nên việc đạo hàm chính xác một hàm số là điều tiên quyết, có 1 cách để debug xem mình đã đạo hàm đúng hay chưa dựa trên định nghĩa của đạo hàm. Tham khảo thêm [kiểm tra đạo hàm](https://machinelearningcoban.com/2017/01/12/gradientdescent/#ki%e1%bb%83m-tra-%c4%91%e1%ba%a1o-h%c3%a0m).

<a name="6-references"></a>

## 6. Tham khảo

[1] Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly.

[2] [Week 2 - Machine Learning coursera by Andrew Ng](https://www.coursera.org/learn/machine-learning/lecture/6Nj1q/multiple-features) 

[3] [Bài 7: Gradient Descent (phần 1/2) - Machine Learning cơ bản by Vu Huu Tiep](https://machinelearningcoban.com/2017/01/12/gradientdescent/)

[4] [Gradient descent by Dung Lai](https://dunglai.github.io/2017/12/21/gradient-descent/)