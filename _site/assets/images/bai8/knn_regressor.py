import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

y = np.array([[2,5,7,9,11,16,19,23,25,29,29,35,37,35,40,42,39,31,30,28,20,15,10]],dtype='float').T
X = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]],dtype='float').T

x0 = np.linspace(2,24,600).reshape(-1,1)
K = 3

knn = neighbors.KNeighborsClassifier(n_neighbors = K)
knn.fit(X,y)
y_predict = knn.predict(x0)

plt.plot(x0,y_predict)
plt.plot(X,y,'ro')
plt.title('K = %d, weight = uniform' %K)
plt.show()