import cv2
import numpy as np
import matplotlib.pyplot as plt

gray_img = cv2.imread('test.png',0)
def gaussian_kernel(kernel_size, sigma=1):
    """
    Parameters
    ----------
    kernel_size : odd number
    sigma : standard deviation

    Returns
    -------
    g : kernel matrix (sliding window)

    """
    size = int(kernel_size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolute(X, kernel, stride = 1, padding = 0):
    """

    Parameters
    ----------
    X : Input image
    kernel : kernel matri

    Returns
    -------
    Y : Convoluted image

    """
    
    x_h,x_w = X.shape
    k_h,k_w = kernel.shape
    
    height = (x_h + 2*padding - k_h)//stride + 1 
    width = (x_w + 2*padding - k_w)//stride + 1
    
    Y = np.zeros((height,width))
    
    for i in range(0,x_h-k_h,stride):
      for j in range(0,x_w-k_w,stride):
          
        if (i<height and j<width):
          Y[i,j] = np.sum(np.multiply(X[i:i+k_h,j:j+k_w],kernel))
          
    return Y

def sobel(image):
    """
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    G : Gradient image
    theta : Angle image (radian type)

    """
    f_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])
    
    f_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
    G_x,G_y = convolute(image,f_x,1,1),convolute(image,f_y,1,1)
    
    G = np.hypot(G_x,G_y) # sqrt(G_x**2 + G_y**2)
    G = G / G.max() * 255 # normalize range
    theta = np.arctan2(G_y, G_x)

    return G,theta
def non_max_suppression(img, theta):
    """
    Parameters
    ----------
    img : Gradient image
    theta : Angle image

    Returns
    -------
    Z : nonMax image

    """
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z