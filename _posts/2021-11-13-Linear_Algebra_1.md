---
layout: post
author: dinhhuyhoang
title: Bài 4 - Ôn tập đại số tuyến tính (1/2)
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
- [4. Ma trận](#4-matrix)
    - [4.1. Tích giữa 2 ma trận](#41-multiply)
    - [4.2. Các ma trận đặc biệt](#42-special)
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

Để khai báo vector trong numpy, ta khai báo chúng trong ngoặc vuông như sau:

```python
v = np.array([0,1.1,2.2,3,4]) # Khai báo vector
print(v)
>>> [0,1.1,2.2,3,4]
```

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

-- Cộng: $\textbf{u} + \textbf{v} = \textbf{r}$, trong đó $r_i = u_i/s_i$. . Cộng 2 vector theo quy tắc hình bình hành:

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

<a name="4-matrix"></a>

## 4. Ma trận

Ma trận hiểu đơn giản nó sẽ ghép nhiều vector để thành một ma trận. Việc một vector chỉ lưu trữ được một biến phụ thuộc, nhưng ma trận sẽ lưu trữ được nhiều biến hơn. Kí hiệu: $\mathbf{A} \in R^{m \times n}$ - tức ma trận A gồm $m$ hàng và $n$ cột.

$$ \begin{split}\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}\end{split} $$

Trong numpy, ma trận sẽ là một mảng 2 chiều. Vì vậy để truy xuất một phần tử, ta sử dụng $\mathbf{A_{ij}}$ trong đó $i$ chỉ vị trí hàng, $j$ chỉ vị trí cột. Để lấy toàn bộ hàng $i$, sử dụng: $\mathbf{A_{i:}}$ và lấy toàn bộ cột $j$, sử dụng: $\mathbf{A_{:j}}$. Thao tác với numpy

```python
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [-1,0,1]])

# Chiều của ma trận A
print("Shape of A: ", A.shape)
>>> Shape of A: (4, 3)

# Truy xuất phần tử hàng 2, cột 3
# Trong numpy, vị trí hàng và cột bắt đầu từ 0
print("A[2][3] = ", A[1][2])
>>> A[2][3] = 6

# Lấy toàn bộ hàng 1
print("Hàng 1: ", A[:1,:])
>>> Hàng 1: [[1, 2, 3]]

# Lấy toàn bộ cột 2
print("Cột 2: ", A[:,1:2])
>>> Cột 2:  [[2], [5], [8],[0]]
```
<a name="41-multiply"></a>

### 4.1. Các phép tính trên ma trận

Hai ma trận có cùng kích thước chúng ta có thể thực hiện các phép cộng, trừ, tích hadamard (hoặc elementi-wise) bằng việc sử dụng broadcasting tương tự với phần [3.2.2](https://hnhoangdz.github.io/2021/11/13/Linear_Algebra_1.html#322-vector-vector). 

- **Tích hadamard hoặc elementi-wise**: Cho $\mathbf{A} \in \mathbb{R}^{m \times n}$ và $\mathbf{B} \in \mathbb{R}^{m \times n}$. 

$$
\begin{split}\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}\end{split}
$$

- **Tích thông thường**: Cho 2 ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và $\mathbf{B} \in \mathbb{R}^{n \times p}$, tích của 2 ma trận được kí hiệu: $\mathbf{C} = \mathbf{A} \mathbf{B} \in \mathbb{R}^{m \times p}$. Lưu ý: để nhân được 2 ma trận này, ta cần số cột của $\mathbf{A}$ bằng số hàng của $\mathbf{B}$, ví dụ:

$$\mathbf{A} = \begin{bmatrix} a_{11} && a_{12} \\ a_{21} && a_{22} \\a_{31} && a_{32} \end{bmatrix} \in \mathbb{R}^{3\times2}, \mathbf{B} = \begin{bmatrix} b_{11} && b_{12} \\ b_{21} && b_{22} \end{bmatrix} \in \mathbb{R}^{2\times2}$$

$$\mathbf{C} = \mathbf{A} \mathbf{B} =  \begin{bmatrix}a_{11}b_{11} + a_{12}b_{21} && a_{11}b_{12} + a_{12}b_{22}
\\ a_{21}b_{11} + a_{22}b_{21} && a_{21}b_{12} + a_{22}b_{22} \\ a_{31}b_{11} + a_{32}b_{21} && a_{31}b_{12} + a_{32}b_{22}\end{bmatrix} \in \mathbb{R}^{3\times2} $$

- **Thao tác với python**

```python
A = np.array([[1, 2, 3], 
              [3, 2, 1]])

B = np.array([[2, 1, 2], 
             [1, 3, 0]])

# Element-wise multiplication
print("Element-wise multiplication: ",A*B)
>>> Element-wise:  [[2 2 6]
                    [3 6 0]]

X = np.array([[1, 2, 3], 
              [3, 2, 1]])

Y = np.array([[2, 1], 
              [1, 3],
              [1, 1]])

# Dot multiplication
print("Dot multiplication: ",X.dot(Y))
>>> Dot multiplication:  [[ 7 10]
                          [ 9 10]]
```

<a name="42-special"><a/>

### 4.2. Các ma trận đặc biệt

- **Ma trận chuyển vị (transpose matrix)**: Cho ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và $\mathbf{B} \in \mathbb{R}^{n \times m}$. Ta gọi ma trận B là ma trận chuyển vị của ma trận A nếu $\mathbf{A_{ij}} = \mathbf{B_{ji}}$, trong đó ∀1 ≤ i ≤ n, 1 ≤ j ≤ m. Kí hiệu: $\mathbf{A}^T = \mathbf{B}$. Nếu $\mathbf{A} = \mathbf{A}^{T}$ thì ta gọi $\mathbf{A}$ là ma trận đối xứng (symetric matrix).

$$\begin{split}\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}\end{split} => \begin{split}\mathbf{A}^T=\begin{bmatrix} a_{11} & a_{21} & \cdots & a_{m1} \\ a_{12} & a_{22} & \cdots & a_{m2} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{2n} & \cdots & a_{mn} \\ \end{bmatrix}\end{split}$$

- **Ma trận vuông (square matrix)**: Ma trận vuông là ma trận có số hàng bằng số cột. Cho $\mathbf{A} \in \mathbb{R}^{m \times n}$ thì $m = n$.

- **Ma trận đơn vị (identity matrix)**: là ma trận vuông và có các phần tử nằm trên đường chéo chính bằng 1 và các phần tử còn lại bằng 0. Kí hiệu: $\mathbf{I_n} \in  \mathbb{R}^{n \times n}$, ví dụ $\mathbf{I_3}$:

$$\begin{split}\mathbf{I}_3=
\begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 1 & 0 \\ 
0 & 0 & 1 
\end{bmatrix}
\end{split}$$

- **Ma trận đường chéo (diagonal matrix)**: Là ma trận có các phần tử trên đường chéo chính khác 0 và các phần tử còn lại bằng 0. Ma trận chéo có thể là ma trận không vuông, ví dụ:

$$\begin{split}\mathbf{A}=
\begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 2 & 0 \\ 
0 & 0 & 3 
\end{bmatrix}
\end{split}$$

- **Ma trận tam giác (triangle matrix)**: Một ma trận vuông được gọi là ma trận tam giác trên (upper triangular matrix) nếu tất cả
các thành phần nằm phía dưới đường chéo chính bằng 0 và trường hợp ngược lại là ma trận tam giác dưới, ví dụ về ma trận tam giác trên:

$$\begin{split}\mathbf{A}=
\begin{bmatrix} 
1 & 3 & 4 \\ 
0 & 2 & 5 \\ 
0 & 0 & 3 
\end{bmatrix}
\end{split}$$

- **Ma trận nghịch đảo (inverse matrix)**: Cho ma trận vuông $\mathbf{A} \in \mathbb{R}^{n \times n}$, nếu tồn tại ma trận vuông $\mathbf{B} \in \mathbb{R}^{n \times n}$ mà $\mathbf{A}\mathbf{B} = \mathbf{I_n}$ thì $\mathbf{A}$ là ma trận khả nghịch và $\mathbf{B}$ là ma trận nghịch đảo của $\mathbf{A}$. [Đọc thêm](https://phamdinhkhanh.github.io/deepai-book/ch_algebra/appendix_algebra.html#ma-tran-nghich-dao)

- **Thao tác với numpy** 

```python
A = np.array([[1, 2, 3], 
              [0, 2, 1],
              [4, 5, 6]])

# Ma trận chuyển vị
print("Tranpose of A: ",A.T)
>>> Tranpose of A:  [[1 0 4]
                     [2 2 5]
                     [3 1 6]]

# Ma trận đơn vị
identity_matrix = np.identity(3)
print("Identity matrix: ",identity_matrix)
>>> Identity matrix:  [[1. 0. 0.]
                       [0. 1. 0.]
                       [0. 0. 1.]]

# Ma trận nghịch đảo
A_inv = np.linalg.inv(A)
print("Inverse of A: ",A_inv)
>>> Inverse of A:  [[-0.77777778 -0.33333333  0.44444444]
                    [-0.44444444  0.66666667  0.11111111]
                    [ 0.88888889 -0.33333333 -0.22222222]]
```

<a name="5-evaluation"></a>

## 5. Đánh giá và kết luận

- Trên đây là những kiến thức cơ bản nhất của đại số tuyến tính như ma trận và vector.
- Ở phần 2, một số khái niệm chuyên sâu hơn sẽ được trình bày.

<a name="6-references"></a>

## 6. Tham khảo

[1] [Machine Learning cơ bản ebook](https://github.com/tiepvupsu/ebookMLCB)

[2] [Deep AI ebook](https://phamdinhkhanh.github.io/deepai-book/intro.html)



