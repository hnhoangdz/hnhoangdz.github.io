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

square_sobel,abs_sobel = sobel(gray)
fig = plt.figure(figsize=(30, 30))
rows = 1
columns = 3
imgs = [gray,square_sobel,abs_sobel]
names = ['original','square_sobel','abs_sobel']
for i in range(0,3):
  fig.add_subplot(rows, columns, i+1)
  plt.imshow(imgs[i],cmap='gray')
  plt.axis('off')
  plt.title(names[i])
