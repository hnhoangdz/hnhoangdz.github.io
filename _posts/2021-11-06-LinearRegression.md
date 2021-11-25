---
layout: post
author: dinhhuyhoang
title: Bài 1 - Linear Regression
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Hàm dự đoán](#2-prediction)
- [3. Hàm mất mát](#3-loss)
- [4. Lập công thức chung](#4-generalization)
	- [4.1. Hình học](#41-geometry)
	- [4.2. Đại số tuyến tính](#42-linear-algebra)
- [5. Thực nghiệm với Python](#5-coding)
	- [5.1. Dạng đường thẳng](#51-straight-line)
	- [5.2. Dạng parabol](#52-parabol-line)
- [6. Đánh giá và kết luận](#6-evaluation)
- [7. Tham khảo](#7-references)

<a name="1-introduction"></a>

## 1. Giới thiệu

Hồi quy tuyến tính (Linear Regression) là một thuật toán căn bản nhất đối với bất kì ai bắt đầu học về AI đều sẽ đi qua. Trong thực tế, bài toán hồi quy tuyến tính được ứng dụng rất nhiều vì tính dễ dàng mô tả và dễ dàng triển khai.

Hồi quy nói chung là lớp bài toán thuộc học có giám sát (supervised-learning). Dựa trên dữ liệu có sẵn (tức giá trị mục tiêu đã biết) và sự phụ thuộc của giá trị đầu vào để dự đoán một giá trị mới.

Ví dụ đơn giản nhất như: dựa trên diện tích nhà để đưa ra giá dự đoán, dựa trên chiều cao để dự đoán cân nặng,... hoặc rằng buộc thêm như dựa vào diện tích nhà, số phòng, view để đưa ra giá nhà dự đoán.

<img src="/assets/images/bai1/linearregression.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Ví dụ dự đoán giá nhà dựa trên diện tích</p>

<a name="2-prediction"></a>

## 2. Hàm dự đoán

Hàm dự đoán hay trong machine learning thường hay gọi là model ám chỉ một hàm số $f$ sẽ là hàm mục tiêu của giá trị cần dự đoán. Một ví dụ đơn giản như dự đoán giá nhà dựa trên số phòng ngủ, diện tích. Vậy đầu vào của bài toán toán sẽ là số phòng ngủ và diện tích, đầu ra sẽ là giá trị dự đoán của căn hộ đó.

Biểu diễn toán học: Cho $$\textbf{r}=[r_1, r_2,...,r_m]$$ là vector biểu diễn cho số lượng phòng ngủ mỗi căn hộ lần lượt là $$r_1,r_2,...,r_m$$, $$\textbf{s}=[s_1, s_2,...,s_m]$$ cũng tương tự $$\textbf{r}$$ là vector biểu diễn diện tích. Cuối cùng, $$\textbf{y}=[y_1, y_2,...,y_m]$$ là dữ liệu giá mỗi căn hộ đã có sẵn. Các giá trị này đều đã biết trước, mục tiêu của chúng ta là dự đoán xem khi có một căn hộ có $$\textbf{a}$$ phòng ngủ, diện tích $$\textbf{b}$$ vậy giá căn hộ này là bao nhiêu?

Trong machine learning, ký hiệu $$\hat{y}$$ thường dùng để biểu diễn cho một giá trị dự đoán, ta có:

$$\begin{equation} \hat{y_1}=w_0 + w_1r_1 + w_2s_2 ~~~~ \end{equation} $$

trong đó, $$w_0$$ - bias, $$w_1$$ và $$w_2$$ - trọng số là các giá trị cần tìm, $$r_1$$ và $$s_2$$ là các giá trị đã biết. 

<a name="3-loss"></a>

## 3. Hàm mất mát - Loss function

Khi nhắc đến giá trị dự đoán thì sai số luôn luôn được kèm theo. Sai số tức sự khác biệt giữa giá trị dự đoán và giá trị thực. Sai số càng bé chứng tỏ giá trị dự đoán càng chính xác. Đây cũng là mục tiêu của hàm mất mát nhằm giảm thiểu sai số tối đa nhất có thể.

Loss function chỉ sai số của một điểm dữ liệu còn cost function sẽ chỉ ra trung bình sai số trên toàn tập dữ liệu. Đây là những thuật ngữ cơ bản và rất quan trọng trong machine learning. Thường các thuật toán tối ưu sẽ tối ưu phần này và cũng là phàn tối ưu khó nhất vì yêu cầu kiến thức toán lớn.

Giả sử $$f$$ là giá trị thực của một căn hộ, $$f'$$ là giá trị dự đoán và $$e$$ là sai số khi dự đoán. Điều ta mong muốn là làm sao cho phương trình sau xảy ra:

$$f \approx f' + e$$

Và trong bài toán hồi quy tuyến tính (Linear Regression) chúng ta cần tối ưu sao cho $$e$$ có giá trị nhỏ nhất có thể. Phương trình cần tối ưu trong bài toán:

$$\frac{1}{2}e^2 = \frac{1}{2}(f - f')^2$$

<a name="4-generalization"></a>

## 4. Lập công thức chung

Với bài toán dự đoán giá nhà dựa trên diện tích, ta có:

Hàm dự đoán: 

$$\hat{y_i} = w_0 + w_1x_i$$ 

trong đó $$x_i$$ biểu diễn giá trị diện tích, $$\hat{y_i}$$ biểu diễn giá trị dự đoán tương ứng của mẫu dữ liệu thứ $i$, $w_0$ và $w_1$ là trọng số cần tìm

Hàm mất mát:

$$J = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y_i}-y_i)^2$$

trong đó $m$ là số lượng dữ liệu ($$\textbf{#samples}$$), $y_i$ là giá trị thực của mẫu dữ liệu thứ $i$.

Lúc này khi đã định nghĩa được 2 hàm số trên, chúng ta có thể đi tìm nghiệm cho bài toán tức tìm $w_0$ và $w_1$. Với thuật toán này, có nhiều phương pháp để tìm nghiệm. Các phương pháp dựa trên toán hình học, đại số tuyến tính và giải tích. Trong bài này, mình sẽ trình bày 2 phương pháp: hình học và đại số tuyến tính.

<a name="41-geometry"></a>

### 4.1. Hình học

Cho 3 điểm dữ liệu $A(x_1,y_1)$, $B(x_2,y_2)$, $C(x_3,y_3)$ và đường màu đỏ chính là đường thẳng ta cần tìm, ta có

<img src="/assets/images/bai1/anh1.png" class="normalpic"/>

Giả sử đường thẳng màu đỏ đi qua 3 điểm A,B,C thì ta có 3 phương 2 ẩn $w_0$ và $w_1$:

$$y_1 = w_0 + w_1x_1 (1)$$ 

$$y_2 = w_0 + w_1x_2 (2)$$

$$y_3 = w_0 + w_1x_3 (3)$$

Nhưng với 3 phương trình 2 ẩn thì việc tìm nghiệm chính xác là điều không thể, vì vậy ta sẽ đi tìm $w_0$ và $w_1$ gần đúng sao cho sai số là bé nhất tức $d_A^{2} + d_B^{2} + d_C^{2}$ bé nhất.

Từ (1), (2), (3) ta có thể vector hóa dưới dạng: $$\begin{bmatrix}y_1\\ y_2 \\ y_3\end{bmatrix} \approx w_0\begin{bmatrix}1\\ 1 \\ 1\end{bmatrix} + w_1\begin{bmatrix}x_1\\ x_2 \\ x_3\end{bmatrix} (4)$$. Đặt $$y = \begin{bmatrix}y_1\\ y_2 \\ y_3\end{bmatrix}$$, $$o = \begin{bmatrix}1\\ 1 \\ 1\end{bmatrix}$$,$$x = \begin{bmatrix}x_1\\ x_2 \\ x_3\end{bmatrix}$$. Lúc này biểu diễn dưới dạng hình học ta sẽ được

<img src="/assets/images/bai1/anh3.png" class="normalpic"/>

Với những giá trị khác nhau của $w_0$ và $w_1$ ta sẽ thu được mặt phẳng $\textbf{(P)}$, hơn nữa vế phải của phương trình (4) sẽ tạo được một vector $\textbf{a}$ nằm trong mặt phẳng $\textbf{(P)}$. Do đó, ta cần tìm $w_0$ và $w_1$ để $\textbf{y}$ và $\textbf{a}$ gần nhau nhất. Để điều kiện này xảy ra khi và chỉ khi $\textbf{a}$ chính là hình chiếu của $\textbf{y}$ lên mặt phẳng $\textbf{(P)}$, gọi vector hình chiếu đó là $\textbf{h}$.

Lúc này ta có $\textbf{h}$ $\bot$ $\textbf{x}$ và $\textbf{h}$ $\bot$ $\textbf{o}$ (tính chất của đường thẳng vuông góc mặt phẳng) $=>$ $\textbf{x}^T.\textbf{h}=0$ và $\textbf{o}^T.\textbf{h}=0 (5)$ 

Đặt $$W = \begin{bmatrix}w_0\\ w_1 \end{bmatrix}$$ và $$X = \begin{bmatrix}1 && x_1 \\ 1 && x_2 \\ 1 && x_3 \end{bmatrix}$$, theo (5) thì ta được $X^T.\textbf{h}=0$ mà $\textbf{h}$ = $\textbf{y} - \textbf{a}$ (tính chất cộng vector) 

$$=>X^T(\textbf{y} - \textbf{a}) = 0$$

$$<=>X^T.\textbf{y} - X^T.\textbf{a} = 0$$

Lại có $\textbf{a} = X.W => X^T.\textbf{y} = X^T.X.W => W = (X^T.X)^{-1}.X^T.\textbf{y}$ (Lưu ý: dấu . biểu diễn dot product).

Tới đây chúng ta đã tìm được W là vector chứa 2 giá trị $w_0$ và $w_1$.

<a name="42-linear-algebra"></a>

### 4.2. Đại số tuyến tính

Để cho dễ dàng triển khai code ở phần sau, thì bắt đầu từ phần này các dữ liệu sẽ được vector và ma trận hóa. Vì vậy các công thức liên quan cũng sẽ cho ra vector hoặc ma trận.

Cho $m$ mẫu dữ liệu đã có nhãn về giá trị diện tích và giá trị căn hộ.

$$X = \begin{bmatrix} 1 && x_1 \\ 1 && x_2 \\ ... && ... \\ 1 && x_m \end{bmatrix}, Y = \begin{bmatrix} y_1 \\ y_2 \\ ... \\ y_m \end{bmatrix}, W = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}, \hat{Y} = X.W = \begin{bmatrix} w_0*1 + w_1*x_1  \\ w_0*1 + w_1*x_2 \\ ... \\ w_0*1 + w_1*x_m \end{bmatrix} = \begin{bmatrix} \hat{y_1} \\ \hat{y_2} \\ ... \\ \hat{y_m} \end{bmatrix}$$

$$J = \frac{1}{2m}sum(\hat{Y}-Y)^2$$

Để tìm nghiệm tối ưu bài toán, ta sẽ đạo hàm hàm J để tìm cực tiểu

$$\frac{dJ}{dW} = \frac{1}{m}X^T.(\hat{Y}-Y) = 0$$

$$<=>X^T.(X.W - Y) = 0$$

$$<=>X^T.X.W = X^T.Y$$

$$<=>W = (X^T.X)^{-1}.X^T.Y$$

Ta thấy nghiệm giải bằng phương pháp sử dụng đại số tuyến tính kết hợp đạo hàm cho ra nghiệm bài toán giống với phương pháp hình học trên. Phương pháp này cũng sẽ là cơ sở để chúng ta bắt đầu với các thuật toán tối ưu trong machine learning như Gradient Descent.

<a name="5-coding"></a>

## 5. Thực nghiệm với Python

<a name="51-straight-line"></a>

### 5.1. Dạng đường thẳng

```python
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
```

<img src="/assets/images/bai1/anh5.png" class="normalpic"/>

<p align="center"> <b>Hình 2</b>: Visualize đường thẳng cần tìm</p>

<a name="52-parabol-line"></a>

### 5.2. Dạng Parabol

Khác một chút so với dữ liệu dạng đường thẳng, nếu dữ liệu dạng không đơn điệu thì hàm dự đoán của ta cần thay đổi một chút. Ví dụ với hàm parabol đã quen thuộc khi còn học cấp 3: $y = ax^2 + bx + c$. Do vậy ở bài toán với hàm dự đoán là parabol, ta cần tìm 3 nghiệm $w_0, w_1$ và $w_2$.

```python
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

# lấy điểm đầu và điểm cuối 
# để vẽ đường thẳng cần tìm
x0 = np.linspace(2,25,10000)
y0 = w0 + w1*x0 + w2*(x0**2)

# visualize đường thẳng cần tìm
plt.plot(x0,y0,'r')
plt.xlabel('Diện tích')
plt.ylabel('Giá nhà')
plt.show()
```

<img src="/assets/images/bai1/anh6.png" class="normalpic"/>

<p align="center"> <b>Hình 3</b>: Visualize parabol cần tìm</p>

<a name="6-evaluation"></a>

## 6. Đánh giá và kết luận

- Nếu bài toán có dữ liệu dạng parabol mà vẫn sử dụng hàm dự đoán là đường thẳng thì sao? Vẫn được, nhưng sai số cao, chưa thể tối ưu bằng sử dụng hàm parabol. Giả sử, đầu vào lúc này không phải là 1 chiều mà là 2 chiều thì hàm dự đoán sẽ trở thành một mặt phẳng và công thức nghiệm trên vẫn đúng. Còn đầu vào có quá nhiều chiều dữ liệu thì lúc này khá khó để giải bằng numpy nên ta sẽ sử dụng thư viện sklearn. Như vậy bài toán dự đoán của chúng ta đã được giải quyết bằng cách tìm nghiệm W. Để dự đoán một điểm mới ta chỉ cần áp dụng như tính y0 ở trên. 

- Việc tính ma trận nghịch đảo là điểm yếu của cách làm này vì sẽ tốn thêm time và space complexity.

- Ở hàm mất mát, có một số thuật toán nhằm tránh overfiting như Ridge Regression, Lasso Regression hay trong deep learning là regurlization.

- Thuật toán này sử dụng trung bình tổng sai số của giá trị dự đoán và giá trị thật để triển khai và đánh giá, vì vậy khi tồn tại dữ liệu "nhiễu" sẽ gây ảnh hưởng tới chất lượng dự đoán. Một số phương pháp khắc phục đó là sử dụng: MAE Loss, Huber Loss... Nhưng vì các hàm này khá khó để giải trực tiếp vì vậy, thường sẽ loại bỏ các dữ liệu "nhiễu" trước khi traning.

<a name="7-references"></a>

## 7. Tham khảo

[1] [Machine Learning cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/)

[2] [Dung Lai github page](https://dunglai.github.io/2017/10/10/linear-regression/)