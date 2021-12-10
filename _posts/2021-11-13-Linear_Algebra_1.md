---
layout: post
author: dinhhuyhoang
title: Bài 4 - Ôn tập đại số tuyến tính (1/3)
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Số vô hướng](#2-scalar)
- [3. Vector](#3-vector)
    - [3.1. Các thuộc tính của vector](#31-attributes)
    - [3.2. Các phép tính với vector](#32-operations)
        - [3.2.1 Cộng, trừ, nhân, chia vector với một số](#321-vector-number)
        - [3.2.2 Cộng, trừ, nhân, chia giữa 2 vector](#322-vector-vector)
    - [3.3. Tích vô hướng](#33-dot-product)
    - [3.4. Cosine rule](#34-cosine)
    - [3.5. Hình chiếu](#35-projection)
- [4. Norm](#4-norm)
    - [4.1. Khoảng cách](#41-distance)
- [5. Đánh giá và kết luận](#5-evaluation)
- [6. Tham khảo](#6-references)

<a name="1-introduction"></a>

## 1. Giới thiệu

Đại số tuyến tính là một mảng toán học rất quan trọng trong machine learning, hầu hết các thuật toán đều dựa trên các tính chất và ứng dụng để tìm ra giải pháp. Tuy nhiên, hiện nay machine learning đã có rất nhiều frameworks và thư viện hỗ trợ nên việc build ra một mô hình machine learning trở nên rất dễ dàng. Đây có thể rất tốt đối với những người đã có kiến thức nền tảng toán căn bản, nhưng đối với mình thời gian đầu tiếp cận thì rất dễ dàng nhưng càng đi sâu thì khá khó khăn với việc những thuật toán nâng cao đều liên quan tới toán học. Vì vậy trong bài hôm nay, mình sẽ tóm gọn các nội dung cơ bản của đại số tuyến tính và thao tác với numpy - một thư viện quan trọng trong machine learning.

Ở trong bài nếu biểu diễn vector là các chữ cái in đậm thường như: $\textbf{a}, \textbf{b}$, các số vô hướng là chữ cái in thường: $a, b$ hoặc các ma trận là chữ cái in hoa đậm: $\textbf{A}, \textbf{B}$

<a name="2-scalar"></a>

## 2. Số vô hướng

Số vô hướng (scalar), là giá trị thực không có hướng. Ví dụ số lượng quả táo, cân nặng, chiều cao... Số vô hướng được kí hiệu: $x \in \mathbb{R}$. Để khai báo một số vô hướng với numpy và tính toán:

```python
import numpy as np # ĐSTT

x = np.array(3) # Khai báo số vô hướng x
y = np.array(4) # Khai báo số vô hướng y

print("a + b = ", a + b)
>>> a + b = 7

print("a x b = ", a * b)
>>> a x b = 12
```

<a name="3-vector"></a>

## 3. Vector

Vector thực ra chúng ta đã được học ở cấp 3, như các bài toán tìm vector hình chiếu. Nhưng chưa hiểu rõ ý nghĩa của vector là gì ngoài biết nó có phương, chiều, độ dài và độ lớn. Vector là một khái niệm rất căn bản, nó có thể biểu diễn bất kì đại lượng nào trong thực tế, ví dụ như: diện tích các căn nhà, tuổi của các học sinh trong lớp...

Vector thường được biểu diễn dưới dạng cột array và có ngoặc vuông bao quanh, ví dụ:

$$\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ -1  \end{bmatrix} $$

Vector thường được kí hiệu như sau: $\mathbf{v} \in \mathbb{R}^n$, trong đó $n$ là số lượng phần tử (dimension, length, size) của vector $\mathbf{v}$. Với ví dụ trên $ n = 3$.

Những giá trị bên trong vector còn được gọi là: element, entries, coefficients, components. Để lấy ra giá trị $ith = v_i$ bên trong vector, ta có thể truy xuất tương tự như truy xuất phần tử trong mảng hoặc list (lưu ý: index của vector trong ngôn ngữ lập trình bắt đầu từ $0$ tới $n-1$, trong toán học $1$ tới $n$).

**Thao tác với numpy**

```python
v = np.array([0,1.1,2.2,3,4]) # Khai báo vector
print(v)
>>> [0,1.1,2.2,3,4]

print(v[2]) # truy xuất phần tử thứ 2
>>> 2.2
```

**Vector đơn vị (unit vector)**

Các phần tử bên trong chỉ có duy nhất một phần tử có giá trị bằng 1, còn lại đều bằng 0. Kí hiệu: với vector $\mathbf{e_i}$, giá trị tại phần tử thứ $i$th $=1$, ví dụ:

$$\mathbf{e_1} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \mathbf{e_2} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix},\mathbf{e_3} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

<a name="31-attributes"></a>

### 3.1. Các thuộc tính của vector

Với việc khai báo vector bằng numpy, ta có thể dễ dàng lấy được một số thuộc tính quan trọng như: độ dài, kiểu dữ liệu, tổng, trung bình, giá trị lớn nhất, giá trị nhỏ nhất bằng cách:

```python
# Độ dài của vector
print("Length of v: ", len(v))
>>> Length of v: 5

# Data type
print("Data type of vector: ",v.dtype)
>>> Data type of vector: float64

# Tổng các giá trị 
print("Sum of v: ", v.sum())
>>> Sum of v: 10.3

# Giá trị nhỏ nhất
print("Min of v: ", v.min())
>>> Min of v: 0.0

# Giá trị lớn nhất
print("Max of v: ", v.max())
>>> Max of v: 4.0

# Giá trị trung bình
print("Mean of v: ", v.mean())
>>> Mean of v: 2.06

# Độ lệch chuẩn
print("std of v: ", v.std())
>>> std of v: 1.4022838514366482
```

<a name="32-operations"></a>

### 3.2. Các phép tính với vector

<a name="321-vector-number"></a>

#### 3.2.1 Cộng, trừ, nhân, chia vector với một số

Với numpy việc tính toán rất dễ dàng, đó là một ứng dụng broadcasting. Cho phép cộng, trừ, nhân, chia trực tiếp. Thao tác với numpy 

```python
v = np.array([0,1.1,2.2,3,4])
k = 5

# Cộng
print("v + k = ",v+k)
>>> v + k = [5., 6.1, 7.2, 8., 9.]

# Trừ
print("v - k = ",v-k)
>>> v - k = [-5., -3.9, -2.8, -2., -1.]

# Nhân
print("v * k = ",v*k)
>>> v * k =  [0., 5.5, 11., 15., 20.]

# Chia
print("v / k = ",v/k)
>>> v / k =  [0., 0.22, 0.44, 0.6, 0.8]
```
<a name="322-vector-vector"></a>

#### 3.2.2 Cộng, trừ, nhân, chia giữa 2 vector

Cho 2 vector $\textbf{u}, \textbf{v} \in \mathbb{R}^{n}$. Lưu ý: 2 vector cần phải cùng độ dài (cùng chiều).

-- Cộng: $\textbf{u} + \textbf{v} = \textbf{r}$, trong đó $r_i = u_i + s_i$. . Cộng 2 vector theo quy tắc hình bình hành:

<img src="/assets/images/bai4/anh2.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Tổng 2 vector </p>

-- Trừ: tương tự phép cộng

-- Nhân: Với numpy ta có thể nhân element-wise trực tiếp giữa 2 vector. $\textbf{s} = \textbf{u} * \textbf{v}$ trong đó $s_i = u_i*s_i$. 

-- Chia: Ta có thể chia trực tiếp 2 vector. $\textbf{s} = \textbf{u} / \textbf{v}$ trong đó $s_i = u_i/s_i$. 

```python
u = np.array([1, 3, 2, 1.8, 1.9])
v = np.array([1, 2.2, 1.2, 1.6, 1.7])

# Cộng
print("u + v: ", u + v)
>>> u + v:  [2., 5.2, 3.2, 3.4, 3.6]

# Trừ
print("u - v: ", u - v)
>>> u - v:  [0., 0.8, 0.8, 0.2, 0.2]

# Nhân
print("u * v: ", u * v)
>>> u * v:  [1., 6.6, 2.4, 2.88, 3.23]

# Chia
print("u / v: ", u / v)
>>> u / v:  [1., 1.36363636, 1.66666667, 1.125, 1.11764706]
```

<a name="33-dot-product"></a>

### 3.3. Tích vô hướng

Tích vô hướng hay còn một số tên gọi khác như: dot product, inner product, scalar product. Tích vô hướng được sử dụng khá nhiều trong machine learning. Tích vô hướng giữa hai vector $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{n}$ có cùng kích thước là một số vô hướng được ký hiệu là $\langle \mathbf{u}, \mathbf{v} \rangle$ có công thức như sau:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^{n} u_i*v_i$$

Ngoài ra để tính độ lớn của vector $\mathbf{r} \in \mathbb{R}^{n}$ ta có thể sử dụng: $\lvert r \rvert  = (\sum_{i=1}^{n} r_i^{2})^{\frac{1}{2}} = \langle \mathbf{r}, \mathbf{r} \rangle^{\frac{1}{2}}$. Thao tác với numpy:

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# dot product
print("<u,v> = ", u.dot(v))
>>> <u,v> = 32

# tính độ lớn của vector
print("Size of u: ",np.sqrt(u.dot(u)))
>>> Size of u: 3.7416573867739413
```

Lưu ý: từ phần này trở đi, tích vô hướng của 2 vector sẽ biểu diễn bằng dấu "." và tích element-wise là dấu "*".

Một số tính chất quan trọng:

- $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$

- $\mathbf{u}.\mathbf{v} = \mathbf{v}.\mathbf{u}$

- $\mathbf{u}.(\mathbf{t}.\mathbf{v}) = \mathbf{u}.\mathbf{t} + \mathbf{u}.\mathbf{v}$

- $\mathbf{u}.(\alpha \mathbf{v}) = \alpha (\mathbf{u}.\mathbf{v})$
<a name="34-cosine"></a>

### 3.4. Cosine rule

Cho tam giác ABC với độ dài 3 cạnh: AC = a,AB = b, AC = c và 3 vector tương ứng 3 cạnh: $\mathbf{u},\mathbf{v},\mathbf{u-v}$. Góc $\alpha$ là góc tạo bởi AB và AC.

<img src="/assets/images/bai4/anh3.png" class="normalpic"/>

<p align="center"> <b>Hình 2</b>: Tam giác ABC </p>

Theo định lý hàm cosin ta có: $c^2 = a^2 + b^2 - 2ab\cos{\alpha}$

$$=> \lvert \mathbf{u-v} \rvert^2 = \lvert \mathbf{u} \rvert^2 + \lvert \mathbf{v} \rvert^2 - 2\lvert \mathbf{u} \rvert \lvert \mathbf{v} \rvert\cos{\alpha} (1)$$

Mà $$ \lvert \mathbf{u-v} \rvert^2 = (\mathbf{u-v}).(\mathbf{u-v}) =  \mathbf{u}.\mathbf{u} +  \mathbf{v}.\mathbf{v} - 2\mathbf{u}.\mathbf{v} = \lvert \mathbf{u} \rvert^2 + \lvert \mathbf{v} \rvert^2 - 2\mathbf{u}.\mathbf{v} (2)$$

Từ (1) và (2) $ => \cos{\alpha} = \frac{\mathbf{u}.\mathbf{v}}{\lvert \mathbf{u} \rvert \lvert \mathbf{v} \rvert}$. Suy ra nếu 2 vector vuông góc, tích vô hướng bằng 0.

Việc tính cosine giữa 2 vector có thể so sánh được sử dụng để so sánh độ tương đồng giữa 2 đại lượng.

<a name="35-projection"></a>

### 3.5. Hình chiếu

Cho 2 vector: $\mathbf{u}, \mathbf{v} $. Góc $\alpha$ là góc tạo bởi 2 vector trên. Vector $\mathbf{h}$ là vector hình chiếu của $\mathbf{u}$ xuống $\mathbf{v}$. Tìm vector $\mathbf{h}$ và độ lớn của $\mathbf{h}$.

<img src="/assets/images/bai4/anh4.png" class="normalpic"/>

<p align="center"> <b>Hình 3</b>: Hình chiếu </p>

Ta có: 

$$\cos{\alpha} = \frac{\lvert \mathbf{h}\rvert}{\lvert \mathbf{u} \rvert} =  \frac{\mathbf{u}.\mathbf{v}}{\lvert \mathbf{u} \rvert \lvert \mathbf{v} \rvert} => \lvert \mathbf{h}\rvert = \frac{\mathbf{u}.\mathbf{v}}{\lvert \mathbf{u} \rvert} $$

$$=> \mathbf{h} = \frac{\mathbf{u}.\mathbf{v}}{\lvert \mathbf{u} \rvert} . \frac{\mathbf{u}}{\lvert \mathbf{u} \rvert}$$

<a name="4-norm"></a>

## 4. Norm

Trong machine learning, đôi khi chúng ta cần tính toán độ lớn (size) của một vector. Để tính toán độ lớn đó, ta sẽ sử dụng khái niệm về $\mathbf{norm}$. Công thức được đưa ra như sau:

$$ \|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p} (1)$$

trong đó $p \in \mathbf{R}, p \geq 1$.

Norm của vector $\mathbf{x}$ còn có thể được gọi là khoảng cách từ gốc tọa độ tới tọa độ của vector $\mathbf{x}$. Một số tính chất về norm:

- $$ \|\mathbf{x}\| \geq 0 $$, dấu bằng xảy ra khi và chỉ khi $\mathbf{x}$ = 0.

- $$\alpha \|\mathbf{x}\| = \|\mathbf{\alpha x}\|$$, việc phóng đại lên véc tơ $\alpha$ lần thì giá trị chuẩn của nó cũng phóng đại lên $\alpha$ lần.

- $$ \|\mathbf{x}\| + \|\mathbf{y}\| \geq \|\mathbf{x + y}\| $$, trong đó vector $\mathbf{y}$ cùng chiều vector $\mathbf{x}$. Bất đẳng thức trong tam giác: tổng 2 cạnh luôn lớn hơn cạnh còn lại.

Với công thức (1) là trường hợp tổng quát nhất của norm ($L^p$), còn có tên gọi là **Minkowski Norm**. Việc thay đổi giá trị $p$ sẽ tạo ra các độ đo khác nhau, một số độ đo thường được sử dụng:

- **Với $p = 2$ - Euclidean Norm ($L^2$)**

$$\|\mathbf{x}\|_2 = \left(\sum_{i=1}^n \left|x_i \right|^2 \right)^{1/2} = (\mathbf{x}^T \mathbf{x})^{1/2}$$

- **Với $p = 1$ - Mahattan Norm ($L^1$)**

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|$$

- **Với $p \rightarrow \infty$ - Chebyshev Norm**

$$||\mathbf{x}||_{\infty} = \max_{i = 1, 2, \dots, n} |x_i|$$

<a name="41-distance"></a>

### 4.1 Khoảng cách

Ở các bài toán về tính khoảng cách giữa 2 điểm bậc THPT, chúng ta đã quá quen thuộc với khoảng cách Euclidean. Khoảng cách này chính là một norm với giá trị $p = 2$. Cho 2 vector $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$d_2(\mathbf{x},\mathbf{y}) = ||\mathbf{x - y}||$$

Với $p = 1$, khoảng cách Mahattan được định nghĩa là norm 1.

$$d_1(\mathbf{x},\mathbf{y}) = ||\mathbf{x - y}||$$

**So sánh sự khác biệt norm 1 và norm 2**:

<img src="/assets/images/bai4/anh5.png" class="normalpic"/>

<p align="center"> <b>Hình 4</b>: Norm 1 và norm 2 trong không gian 2 chiều (Nguồn: <a href="https://machinelearningcoban.com/math/#-norms-chuan"> Machine Learning cơ bản</a>)</p>

Norm 2 (màu xanh lục) là đường "chim bay" giữa 2 vector $\mathbf{x}$ và  $\mathbf{y}$. Norm 1 thường được sử dụng trong tính toán đồ thị khi không có đường thẳng nối trực tiếp giữa 2 điểm vì vậy ta chỉ có cách đi dọc theo cạnh.

**Chứng minh tính chất bất đẳng thức trong tam giác**. Cho tam giác và có các vector tọa độ $\mathbf{a},\mathbf{b},\mathbf{b}$ ta có thể suy ra các độ dài cạnh của tam giác như hình sau:

<img src="/assets/images/bai4/anh6.png" class="smallpic"/>

<p align="center"> <b>Hình 5</b>: tam giác</p>

Ta có:

$$ ||\mathbf{a - b}|| + ||\mathbf{b - c}|| \geq ||\mathbf{a - c}|| $$

$$<=>||\mathbf{a - b}|| + ||\mathbf{b - c}|| \geq ||\mathbf{(a - b) + (b - c)}|$$

<a name="5-evaluation"></a>

## 5. Đánh giá và kết luận

- Trên đây là những kiến thức cơ bản nhất về vector trong đại số tuyến tính.
- Các phép toán của vector có rất nhiều ứng dụng trong machine learning như: so sánh độ tương đồng (dựa trên khoảng cách hoặc cosine), giúp việc biểu diễn các features dễ dàng hơn nhằm mục đích tính toán nhanh hơn,..
- 2 thuật toán ở [bài 6 - K-means](https://hnhoangdz.github.io/2021/11/21/Kmeans.html) và [bài 7 - K-nearest neighbors](http://localhost:4000/2021/11/25/KNN.html) sẽ sử dụng tính chất về norm để giải quyết.
- Ở phần 2, các kiến thức về ma trận và một số khái niệm chuyên sâu sẽ được trình bày.

<a name="6-references"></a>

## 6. Tham khảo

[1] [Machine Learning cơ bản ebook](https://github.com/tiepvupsu/ebookMLCB)

[2] [Deep AI ebook](https://phamdinhkhanh.github.io/deepai-book/intro.html)

[3] [Machine Learning cơ bản](https://machinelearningcoban.com/math/#-norms-chuan)





