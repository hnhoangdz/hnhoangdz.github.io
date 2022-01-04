import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
image = cv2.imread('nature.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
w,h,_ = image.shape
# Transform channels -> vectors
image_vec = np.reshape(image,(-1,3))

# Init K-means 
kmeans = KMeans(n_clusters=3)

# Training
kmeans.fit(image_vec)

# Get labels for each pixel
labels = kmeans.predict(image_vec)

# Get centroids
centroids = kmeans.cluster_centers_

# Init zeros image vec
image_result = np.zeros_like(image_vec)

# Update each pixel equal to corresponding centroids
for i in range(len(image_result)):
    image_result[i] = centroids[labels[i]]

image_result = image_result.reshape(w,h,3)

plt.figure(figsize=(16, 2))
plt.subplot(151)
plt.imshow(image)
plt.title('Origin')
plt.axis(False)


plt.subplot(152)
plt.imshow(image_result)
plt.title('K = 3')
plt.axis(False)