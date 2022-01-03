import numpy as np
import cv2
import matplotlib.pyplot as plt


def otsu_implement(gray):
    # Mean weight to calculate probability
    mean_weight = 1.0/gray.shape[0] * gray.shape[1]
    
    # Histogram and bins edge horizontal axis
    his, bins = np.histogram(gray, np.arange(0,256))

    T = -1 # threshold value
    max_value = -1
    intensity_arr = np.arange(255)
    
    for t in bins[1:-1]:
        
        # Weight/probability of Foreground and Background
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight
        
        # Mean of Foreground and Background
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        
        # Formula must be maximum
        value = Wb * Wf * (mub - muf) ** 2

        if value > max_value:
            T = t
            max_value = value
    gray[gray > T] = 255
    gray[gray <= T] = 0
    return T,gray

def otsu_built(gray):
    (T, threshImg) = cv2.threshold(gray, 0, 255, \
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return T,threshImg

gray_img = cv2.imread('coin.jpg',0)
gray_img2 = gray_img.copy() 

otsu_implement_threshold,thresh_img = otsu_implement(gray_img)
otsu_built_threshold,threshImg = otsu_built(gray_img2)

plt.figure(figsize=(16, 4))
plt.subplot(151)
plt.imshow(gray_img2,cmap='gray')
plt.title('Origin')
plt.axis(False)


plt.subplot(152)
plt.imshow(thresh_img,cmap='gray')
plt.title('T-implement = ' + str(otsu_implement_threshold))
plt.axis(False)


plt.subplot(153)
plt.imshow(threshImg,cmap='gray')
plt.title('T-built = ' + str(otsu_built_threshold))
plt.axis(False)