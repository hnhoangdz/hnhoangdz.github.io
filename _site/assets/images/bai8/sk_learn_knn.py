from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
# np.random.seed(2)
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=50)


knn_clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('Accuracy score: %2f' %acc)