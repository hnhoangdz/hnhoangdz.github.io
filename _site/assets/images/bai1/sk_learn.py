
from sklearn.linear_model import LinearRegression
import numpy as np

# Dữ liệu
X = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
Y = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

# Khởi tạo model
lin_reg = LinearRegression()

# Fit/train model
lin_reg.fit(X, Y)

print('W = {}, b = {}'.format(lin_reg.coef_,lin_reg.intercept_))