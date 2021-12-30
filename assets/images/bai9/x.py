import cv2
import numpy as np
import matplotlib.pyplot as plt
def convolute(X, kernel, stride = 1, padding = 0):
    """
    Parameters
    ----------
    X : Grayscale Image - Numpy Array
    kernel : Numpy Array
    stride : int
    padding : int
    Returns
    -------
    Y : Convoluted Image - Numpy Array

    """
    H,W = X.shape
    k_h,k_w = kernel.shape
    height = (H+2*padding-k_h)//stride+1 
    width = (W+2*padding-k_w)//stride+1
    Y = np.zeros((height,width))
    for i in range(0,H-k_h,stride):
      for j in range(0,W-k_w,stride):
        if (i<height and j<width):
          Y[i,j] = np.sum(np.multiply(X[i:i+k_h,j:j+k_w],kernel))
    return Y

img = cv2.imread('ronaldo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.float32)/25
imgSmooth = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(imgSmooth),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()