---
layout: post
author: dinhhuyhoang
title: 5. Ôn tập đại số tuyến tính (2/3)
---

- [1. Ma trận](#1-matrix)
    - [1.1. Các phép tính trên ma trận](#11-operations)
    - [1.2. Các ma trận đặc biệt](#12-special)
- [2. Định thức](#2-determinants)
- [3. Phép nhân ma trận với vector](#3-matrix_mul_vec)
- [4. Tổ hợp tuyến tính - Không gian sinh](#4-linearcb_span)
- [5. Hệ độc lập tuyến tính](#5-linear_independent)
- [6. Cơ sở của một không gian](#6-basis)
- [7. Biến đổi hệ cơ sở vector](#7-changebasis)
- [8. Hạng của ma trận](#8-rank)
- [9. Norm của ma trận](#9-norm)
- [10. Vết của ma trận](#10-trace)
- [11. Ví dụ](#11-examples)
- [12. Đánh giá và kết luận](#12-evaluation)
- [13. Tham khảo](#13-references)

<a name="1-matrix"></a>

## 1. Ma trận

Ma trận hiểu đơn giản nó sẽ ghép nhiều vector để thành một ma trận. Việc một vector chỉ lưu trữ được một biến phụ thuộc, nhưng ma trận sẽ lưu trữ được nhiều biến hơn. Kí hiệu: $\mathbf{A} \in R^{m \times n}$ - tức ma trận A gồm $m$ hàng và $n$ cột.

$$ \begin{split}\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}\end{split} $$

Trong numpy, ma trận sẽ là một mảng 2 chiều. Vì vậy để truy xuất một phần tử, ta sử dụng $\mathbf{A_{ij}}$ trong đó $i$ chỉ vị trí hàng, $j$ chỉ vị trí cột. Để lấy toàn bộ hàng $i$, sử dụng: $\mathbf{A_{i:}}$ và lấy toàn bộ cột $j$, sử dụng: $\mathbf{A_{:j}}$. 

**Thao tác với numpy**

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
<a name="11-operations"></a>

### 1.1. Các phép tính trên ma trận

Hai ma trận có cùng kích thước chúng ta có thể thực hiện các phép cộng, trừ tương tự như vector.

Nhưng với phép nhân ma trận, có 2 phép tính cần cần lưu ý:

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

- **Tính chất đối với phép nhân thông thường**: 

    - Không có tính giao hoán: $\mathbf{AB} \neq \mathbf{BA}$
    - Tính kết hợp: $\mathbf{(AB)C} = \mathbf{A(BC)} = \mathbf{ABC}$
    - Tính phân phối với phép cộng: $\mathbf{A(B+C)} = \mathbf{AB} + \mathbf{BC}$

**Thao tác với python**

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

<a name="12-special"><a/>

### 1.2. Các ma trận đặc biệt

- **Ma trận chuyển vị (transpose matrix)**: Cho ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và $\mathbf{B} \in \mathbb{R}^{n \times m}$. Ta gọi ma trận B là ma trận chuyển vị của ma trận A nếu $\mathbf{A_{ij}} = \mathbf{B_{ji}}$, trong đó ∀1 ≤ i ≤ n, 1 ≤ j ≤ m. Kí hiệu: $\mathbf{A}^T = \mathbf{B}$. Nếu $\mathbf{A} = \mathbf{A}^{T}$ thì ta gọi $\mathbf{A}$ là ma trận đối xứng (symetric matrix). Một tính chất quan trọng đó là: $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$

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

- **Ma trận nghịch đảo (inverse matrix)**: Cho ma trận vuông $\mathbf{A} \in \mathbb{R}^{n \times n}$, nếu tồn tại ma trận vuông $\mathbf{B} \in \mathbb{R}^{n \times n}$ mà $\mathbf{A}\mathbf{B} = \mathbf{I_n}$ thì $\mathbf{A}$ là ma trận khả nghịch và $\mathbf{B}$ là ma trận nghịch đảo của $\mathbf{A}$. Một tính chất quan trọng của ma trận nghịch đảo đó là: $(\mathbf{AB})^{-1} = \mathbf{A}^{-1} \mathbf{B}^{-1}$

**Thao tác với numpy** 

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
<a name="2-determinant"></a>

## 2. Định thức

Định thức (determinant) của một ma trận $\mathbf{A} \in \mathbb{R}^{n \times n}$ vuông cấp $n$

$$\mathbf{A}=\begin{bmatrix} 
a_{11} & a_{12} & \dots & a_{1n} \\ 
a_{21} & a_{22} & \dots & a_{2n} \\ 
\dots & \dots & \ddots & \dots \\ 
a_{n1} & a_{n2} & \dots & a_{nn} \\ 
\end{bmatrix}$$

Với $n = 1$, $\det(\mathbf{A})$ bằng chính phần tử đó.

Với $n > 1$, ta có công thức sau:

$$\det(\mathbf{A})= \sum_{i=1}^{m} (-1)^{i+j} a_{ij} \det(\mathbf{A_{ij}})$$

trong đó $\mathbf{A_{ij}}$ là ma trận thu được bằng cách xóa hàng $i$ và cột $j$ của ma trận $\mathbf{A}$ hay còn gọi là phần bù đại số của $\mathbf{A}$ ứng với phần tử ở hàng $i$, cột $j$. Thông qua định thức, chúng ta có thể biết được hệ các véc tơ dòng (hoặc cột) của một ma trận là độc lập tuyến tính hay phụ thuộc tuyến tính. (Công thức tính định thức trên không phải là công thức tính theo định nghĩa, công thức tính định thức theo định nghĩa dựa trên số hoán vị từ $1 - n$. Tuy nhiên công thức này khá khó diễn đạt và triển khai ở những ma trận to).

**Một số tính chất quan trọng:**

- $\det{(A)} = \det{(A^T)}$
- $\det{(I_n)} = 1$
- $\det{(AB)} = \det{(A)}\det{(B)}$
- $\det{(A^{-1})} = \frac{1}{\det{(A)}}$
- Để ma trận $\mathbf{A}$ là khả nghịch khi và chỉ khi  $\det{(A)} \neq 0$. Trong trường hợp không tồn tại ma trận  $\mathbf{A}$  thoả mãn điều kiện $\mathbf{AB} = \mathbf{I_n}$ thì ta nói rằng ma trận  $\mathbf{A}$  không khả nghịch hoặc bị suy biến (singular).
- Nếu ma trận có một hàng hoặc một cột là vector 0 thì định thức bằng 0.
- Định thức của một ma trận tam giác (vuông) bằng tích các phần tử trên đường chéo chính.

**Thác tác với numpy**

```python
A = np.array([[1, 2],
              [3, 4]])

det_A = np.linalg.det(A)

print(det_A)
>>> -2.0000000000000004
```
<a name="3-matrix_mul_vec"></a>

## 3. Phép nhân ma trận với vector

Cho ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và vector $\mathbf{x} \in \mathbb{R}^n$, ta có:

$$\mathbf{y} = \mathbf{Ax} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix} \mathbf{x} = \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{x} \\
\mathbf{a}^\top_{2} \mathbf{x} \\
\vdots \\
\mathbf{a}^\top_m \mathbf{x} \\
\end{bmatrix} = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix} \in \mathbb{R}^m$$

Một trong những phép tính được sử dụng rất nhiều đó là dot product của ma trận và vector. Phép tính này thường được áp dụng vào những layer cuối của mạng nơ-ron và có tên gọi là fully connected nhằm giảm chiều giữ liệu. Bản chất của phép tính này là một phép biến hình của vector ban đầu sang một hệ không gian mới. Các phép xoay ảnh cũng biến đổi toạ độ các điểm sang không gian mới bằng cách nhân véc tơ toạ độ của chúng với ma trận xoay.

<a name="4-linearcb_span"></a>

## 4. Tổ hợp tuyến tính - Không gian sinh

**Tổ hợp tuyến tính (linear combination)** được định nghĩa như sau. Cho các vector khác 0: $\mathbf{a_1},\mathbf{a_2},...\mathbf{a_m} \in \mathbb{R}^n$ và các số thực $x_1,x_2,...x_m$. Khi đó vector:

$$\mathbf{b} = x_1\mathbf{a_1} + x_2\mathbf{a_2} + ... x_m\mathbf{a_m}$$

được gọi là một tổ hợp tuyến tính của $\mathbf{a_1},\mathbf{a_2},...\mathbf{a_m} \in \mathbb{R}^n$ và các số thực $x_1,x_2,...x_m$ được gọi là các hệ số (coefficients).

Xét ma trận $\mathbf{A} = [\mathbf{a_1},\mathbf{a_2}...\mathbf{a_m}] \in \mathbb{R}^{m \times n}$ và vector $\mathbf{x} = [x_1,x_2...x_n]^T$, ta có thể viết lại:

$$\mathbf{b} = \mathbf{Ax}$$

và $\mathbf{b}$ là một tổ hợp tuyến tính các cột của $\mathbf{A}$. Hơn nữa $\mathbf{b}$ không là duy nhất, có thể tồn tại nhiều vector $\mathbf{x}$ để thỏa mãn phương trình trên.

Tập hợp tất cả các vector $\mathbf{b}$ có thể biểu diễn được như trên là một tổ hợp tuyến tính của các vector khác 0 $\mathbf{a_1},\mathbf{a_2},...\mathbf{a_m} \in \mathbb{R}^n$ được gọi là **không gian sinh (span space)**, kí hiệu: $span(\mathbf{a_1},\mathbf{a_2},...\mathbf{a_m})$

**Lưu ý:** Tổ hợp tuyến tính của vector đơn vị (unit vector). Vector $\mathbf{x} \in \mathbb{R}^n$ là tổ hợp tuyến tính của các vector đơn vị tạo nên $\mathbf{x}$

$$\mathbf{x} = x_1\mathbf{e_1} + x_2\mathbf{e_2} + ... + x_n\mathbf{e_n}$$

Ví dụ: $$\mathbf{x} = \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix} = 1 \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + (-1) \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} + 2 \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

<a name="5-linear_independent"></a>

## 5. Hệ độc lập tuyến tính

Cho hệ vector {$\mathbf{a_1},\mathbf{a_2},...\mathbf{a_m}$} là các vector khác 0 và các hệ số tương ứng $k_1,k_2,..k_m$, với phương trình:

$$k_1\mathbf{a_1} + k_2\mathbf{a_2} + ... + k_m\mathbf{a_m} = 0$$

- nếu tồn tại duy nhất nghiệm $k_1 = k_2 = ... = k_m = 0$ thỏa mãn phương trình trên thì ta gọi hệ vector trên là độc lập tuyến tính (linear independent). Hay nói cách khác, tồn tại duy nhất một tổ hợp tuyến tính với vector hệ số (coefficients) là 0 của vector 0.

- ngược lại, nếu tồn tại một nghiệm $k_j \neq 0$ thì ta gọi hệ vector trên là phụ thuộc tuyến tính (linear dependent). Thật vậy, lúc này ta chỉ cần chuyển vế với vector $\mathbf{a_j}$ tương ứng với hệ số (coefficients) $k_j$ sẽ thu được một tổ hợp tuyến tính:
 
$$\mathbf{a_j} = (-k_1/k_j)\mathbf{a_1} + (-k_2/k_j)\mathbf{a_2} + ... + (-k_m/k_j)\mathbf{a_m}$$

<a name="6-basis"></a>

## 6. Cơ sở của một không gian

Một hệ các véc tơ {${\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n}$} trong không gian véc tơ $m$ chiều, kí hiệu là $V = \mathbb{R}^{m}$ được gọi là một cơ sở (basis) nếu như điều kiện kiện sau được thoả mãn:

1. $V \equiv \text{span}(\mathbf{a}_1, \dots, \mathbf{a}_n)$

2. ${\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n}$ là một hệ véc tơ độc lập tuyến tính.

Mỗi một vector $\mathbf{b} \in V$ đều có một biểu diễn duy nhất dưới dạng một tổ hợp tuyển tính của những véc tơ của các $\mathbf{a}_i$.

<a name="7-changebasis"></a>

## 7. Biến đổi hệ cơ sở của vector

Trong không gian $n$ chiều, mọi vector có thể biểu diễn thông qua hệ vector đơn vị (unit vector) ($\mathbf{e_1},\mathbf{e_2}...\mathbf{e_n} $). Hơn nữa, việc nhân ma trận với vector được đề cập bên trên cũng sẽ chuyển dịch hệ cơ sở gốc sang một hệ cơ sở mới. Giả sử ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ nhân với vector $\mathbf{x} \in \mathbb{R}^n$

$$\mathbf{Ax} = \mathbf{a}^{(1)} x_1 + \mathbf{a}^{(2)} x_2 + ... + \mathbf{a}^{(n)} x_n = \mathbf{y}$$

trong đó $\mathbf{a}^{(i)}$ là vector cột $i$. Ta có thể xem như véc tơ $\mathbf{y}$ được biểu diễn thông qua các véc tơ cơ sở cột mà toạ độ tương ứng với mỗi chiều trong hệ cơ sở là các $x_i$. 

<a name="8-rank"></a>

## 8. Hạng của ma trận

Xét một ma trận $\mathbf{A} ∈ \mathbb{R}^{m×n}$ . _Hạng_ (_rank_) của ma trận của ma trận được ký hiệu là $\text{rank}(\mathbf{A})$, được định nghĩa là số lượng lớn nhất các cột (hoặc dòng) của nó tạo thành một hệ độc lập tuyến tính.

**Một số tính chất quan trọng:**

- $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A^T})$
- $\text{rank}(\mathbf{A}) \leq min(m,n)$
- $\text{rank}(\mathbf{A}) \leq min(\text{rank}(\mathbf{A}),\text{rank}(\mathbf{B}))$
- $\text{rank}(\mathbf{A + B}) \leq \text{rank}(\mathbf{A}) + \text{rank}(\mathbf{B})$
- Nếu $\mathbf{A} \in \mathbb{R}^{m \times n},\mathbf{B} \in \mathbb{R}^{n \times k} $ thì $\text{rank}(\mathbf{A}) + \text{rank}(\mathbf{B}) - n \leq \text{rank}(\mathbf{AB})$
- Nếu $\mathbf{A}$ là ma trận vuông khả nghịch cấp $n$ thì $\text{rank}(\mathbf{A}) = n$

**Thao tác với numpy**

```python
x = np.array([[1,2,3],
              [4,5,6]])
print(np.linalg.matrix_rank(x))
>>> 2
```

<a name="9-norm"></a>

## 9. Norm của ma trận

Với một ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$, norm được dùng nhiều nhất là Frobenius, kí hiệu $$\|\mathbf{X}\|_F$$ là căn bậc 2 của tổng bình phương các phần tử trong ma trận. 

$$||\mathbf{A}||_F = \sqrt{\sum_{i = 1}^m \sum_{j = 1}^n a_{ij}^2}$$

<a name="10-trace"></a>

## 10. Vết của ma trận

Vết của một ma trận vuông là tổng tất cả các phần tử trên đường chéo chính của nó, được ký hiệu là _trace_($\mathbf{A}$).

**Một số tính chất quan trọng:**

- _trace_($\mathbf{A}$) = _trace_($\mathbf{A^T}$)

- _trace_($\mathbf{AB}$) = _trace_($\mathbf{BA}$)

- Nếu $\mathbf{X}$ là một ma trận khả nghịch và có cùng chiều với ma trận vuông $\mathbf{A}$ thì:
    -  _trace_($\mathbf{XAX^{-1}}$) = _trace_($\mathbf{X^{-1}XA}$) = _trace_($\mathbf{A}$)

- $$\|\mathbf{A}\|_F$$ = _trace_($\mathbf{A^TA}$) = _trace_($\mathbf{AA^T}$) với $\mathbf{A}$ là ma trận bất kì

- _trace_($\mathbf{A}$) = $\sum_{i=1}^n \lambda_i$ với $\lambda_i$ là [trị riêng]() của ma trận $\mathbf{A}$

<a name="11-examples"></a>

## 11. Ví dụ

Cho $\mathbf{A}, \mathbf{B}, \mathbf{C}$ là ba ma trận có kích thước lần lượt là $m \times n$, $n \times p$ và $p \times q$.

**Chứng minh**: $\mathbf{ABC} = (\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ 

- Theo quy ước của Einstein, tích 2 ma trận $\mathbf{A}^{m \times n}$ với $\mathbf{B}^{n \times p}$ được ma trận $\mathbf{C}^{m \times p}$. Ta có:

$$\mathbf{C_{ij}} = \sum_{k = 1}^n \mathbf{A_{ik} \mathbf{B_{kj}}}$$

- Ta thấy tích 3 ma trận $\mathbf{ABC} \in \mathbb{R}^{m \times q}$ vì vậy để chứng minh $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ ta cần chiều (shape) của 2 vế là như nhau (1), giá trị tại hàng $i$ và cột $j$ của 2 vế bằng nhau:

$$((\mathbf{A}\mathbf{B})\mathbf{C})_{ij} = \sum_{k = 1}^p (\mathbf{AB})_{ik} \mathbf{C}_{kj} = \sum_{k = 1}^p \sum_{l = 1}^n \mathbf{A}_{il} \mathbf{B}_{lk}\mathbf{C}_{kj} $$

$$= \sum_{l = 1}^n \mathbf{A}_{il} (\sum_{k = 1}^p \mathbf{B}_{lk}\mathbf{C}_{kj}) = \sum_{l = 1}^n \mathbf{A}_{il} (\mathbf{BC})_{lj} = (\mathbf{A}(\mathbf{BC}))_{ij} (2) $$

- Từ (1) và (2) $=>$ điều phải chứng minh

**Chứng minh**: $$(\mathbf{A}\mathbf{B})^{\intercal} = \mathbf{B}^{\intercal}\mathbf{A}^{\intercal}$$

Chứng minh cách tương tự câu trên, ta có: 

$$((\mathbf{A}\mathbf{B})^{\intercal})_{ij} = (\mathbf{A}\mathbf{B})_{ji} = \sum_{k = 1}^n \mathbf{A}_{jk} \mathbf{B}_{ki} = \sum_{k = 1}^n \mathbf{B}_{ki} \mathbf{A}_{jk} = (\mathbf{B}^{\intercal} \mathbf{A}^{\intercal})_{ij} $$

**Chứng minh**: $\mathbf{A(B+C)} = \mathbf{AB} + \mathbf{BC}$ (lưu ý: phần này ma trận $\mathbf{B}$ và $\mathbf{C}$ có cùng chiều $n \times p$)

$$(\mathbf{A(B+C)})_{ij} = \sum_{k=1}^p \mathbf{A}_{ik} \mathbf{(B+C)}_{kj} =  \sum_{k=1}^p (\mathbf{A}_{ik}\mathbf{B}_{kj} + \mathbf{A}_{ik}\mathbf{C}_{kj}) = (\mathbf{AB} + \mathbf{BC})_{ij}$$

<a name="12-evaluation"></a>

## 12. Đánh giá và kết luận

<a name="13-references"></a>

## 13. Tham khảo






