---
layout: post
author: dinhhuyhoang
title: Bài 1 - Linear Regression
---

**Phụ lục:**

- [1. Giới thiệu](#1-introduction)
- [2. Least Square Approximation and Cost Function](#2-least-square-approximation-and-cost-function)
- [3. Solving Optimization Problem](#3-solving-optimization-problem)
	- [3.1. By Geometry](#31-by-geometry)
	- [3.2. By Algebra](#32-by-algebra)
	- [3.3. By Calculus](#33-by-calculus)
- [4. Discussion](#4-discussion)
	- [4.1. Fitting a parabola](#41-fitting-a-parabola)
- [5. Coding with Python](#5-coding-with-python)
- [6. Further topics to study](#6-further-topics-to-study)
- [7. Reference](#7-reference)

# 1. Giới thiệu

Hồi quy tuyến tính

# 2. Thuật toán

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# create random data 
A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

#Visualize data 
plt.plot(A, b, 'ro')

# append column ones to matrix A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((ones,A), axis=1)

# calculate coefficients to fit a straight line
A_dagger = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose())
x = A_dagger.dot(b)

# start and end point of the straight line
x0 = np.linspace(1,46,2)
y0 = x[0][0] + x[1][0]*x0

# plot the straight line
plt.plot(x0, y0)
plt.xlabel('x coordinates ')
plt.ylabel('y coordinates ')
plt.show()
```
Using Scikit learn library
```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model

# create random data 
A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

#Use Scikit Learn
lr = linear_model.LinearRegression()
lr.fit(A,b)
print('Solution found by scikit learn: w =')
print(lr.intercept_) #w0
print(lr.coef_) #w1
```

Fitting a prabole:

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# create random data 
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

#Visualize data 
plt.plot(A, b, 'ro')

# append column ones to matrix A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((ones,A), axis=1)

## append x^2 to A
x_square = np.array([A[:,1]**2]).T
A = np.concatenate((A,x_square), axis=1)

# calculate coefficients to fit a straight line
A_dagger = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose())
x = A_dagger.dot(b)

# start and end point of the straight line
x0 = np.linspace(1,25,10000)
y0 = x[0][0] + x[1][0]*x0 +x[2][0]*x0*x0

# plot the parabola
plt.plot(x0, y0)
plt.xlabel('x coordinates ')
plt.ylabel('y coordinates ')
plt.show()
```
