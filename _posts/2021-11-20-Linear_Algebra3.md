---
layout: post
author: dinhhuyhoang
title: 6. Ôn tập đại số tuyến tính (3/3)
---

- [1. Hệ trực chuẩn, ma trận trực giao](#1-orthogonal)
- [5. Đánh giá và kết luận](#5-evaluation)
- [6. Tham khảo](#6-references)

## 1. Hệ trực giao, trực chuẩn

Một tập hợp {${\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n}$} $\in \mathbb{R}^{m}$ được gọi là hệ trực giao (orthogonal/mutually orthogonal) nếu mỗi $\mathbf{a_i} \perp \mathbf{a_j}$ với: $i \neq j$ và $i,j \leq n$. Một tập hợp được gọi là hệ trực chuẩn (orthonormal) nếu đó là một hệ trực giao và norm 2 của mỗi vector = 1. Công thức về hệ trực chuẩn được đơn giản hóa như sau:

$$
    \mathbf{a_i}^T \mathbf{a_j}= 
\begin{cases}
    1, &  i = j\\
    0, & i \neq j
\end{cases}$$

Một ví dụ về hệ trực chuẩn đó là tập hợp các vector đơn vị (unit vectors) {${\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n}$}, trong không gian 2D như sau:

<img src="/assets/images/bai6/anh1.png" class="smallpic"/>

<p align="center"> <b>Hình 1</b>: Hệ trực chuẩn</p>

**Hệ trực chuẩn là một hệ độc lập tuyến tính**. Thật vậy, giả sử ta có một tập hợp vector {${\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n}$} là một hệ trực chuẩn và các hệ số tương ứng $k_1,k_2,...k_n$, phương trình:

$$k_1\mathbf{a_1} + k_2\mathbf{a_2} +...+ k_n\mathbf{a_n} = 0 $$

Nhân inner product cả 2 vế với $\mathbf{a_i}$ ta được:

$$\mathbf{a_i}^T(k_1\mathbf{a_1} + k_2\mathbf{a_2} +...+ k_n\mathbf{a_n}) = 0$$

$$k_1(\mathbf{a_i}^T \mathbf{a_1}) + k_2(\mathbf{a_i}^T \mathbf{a_2}) + ... + k_n(\mathbf{a_i}^T \mathbf{a_n}) = 0$$

$$k_i = 0$$

Qua đây ta thấy chỉ tồn tại duy nhất một tập hệ số $k_1 = k_2 = ... = k_n = 0$ là nghiệm cho phương trình trên, vì vậy tổ hợp tuyến tính duy nhất của hệ trực chuẩn là 0. Và hệ trực chuẩn là một hệ độc lập tuyến tính.
