import cv2
import numpy as np
import matplotlib.pyplot as plt

test = cv2.imread('ronaldo.jpg')
gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

def convolute(X, kernel, stride = 1, padding = 0):
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

def normalize_range(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img

def sobel(image):
    f_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])
    
    f_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
    img1,img2 = convolute(image,f_x,1,1),convolute(image,f_y,1,1)
    
    img1 = normalize_range(img1)
    img2 = normalize_range(img2)
    
    square_sobel = np.sqrt(img1**2+img2**2)
    abs_sobel = np.abs(img1) + np.abs(img2)
    return square_sobel,abs_sobel

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

def sobel1(image):
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

def threshold(img, weak_pixel=75, strong_pixel=255, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    """
    Parameters
    ----------
    img : nonMax image
    weak_pixel : if weak pixel
        DESCRIPTION. The default is 75.
    strong_pixel : if strong pixel
        DESCRIPTION. The default is 255.
    lowThresholdRatio : ratio (cái này k bắt buộc vì 
                               trong hàm opencv sẽ cố định một giá trị pixel làm ngưỡng)
        DESCRIPTION. The default is 0.05.
    highThresholdRatio : ratio (cái này k bắt buộc vì 
                                trong hàm opencv sẽ cố định một giá trị pixel làm ngưỡng)
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    res : thresholded image

    """
    maxVal = img.max() * highThresholdRatio
    minVal = maxVal * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    strong_i, strong_j = np.where(img >= maxVal)
    zeros_i, zeros_j = np.where(img < minVal)
    
    weak_i, weak_j = np.where((img <= maxVal) & (img >= minVal))
    
    res[strong_i, strong_j] = strong_pixel
    res[weak_i, weak_j] = weak_pixel
    
    return res

# implement canny
g = gaussian_kernel(5)
convoluted_img = convolute(gray,g)
G,theta = sobel1(convoluted_img)
nonMax_img = non_max_suppression(G, theta)
th = threshold(nonMax_img)

# implement sobel
square_sobel,abs_sobel = sobel(gray)

# sobel built - in
sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = sobelX + sobelY

# canny built - in
k = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(k, 75, 255)
fig = plt.figure(figsize=(30, 30))
rows = 1
columns = 2
imgs = [th,canny]
names = ['Implement Canny','Built-in Canny']
for i in range(0,2):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(imgs[i],cmap='gray')
    plt.axis('off')
    plt.title(names[i])

