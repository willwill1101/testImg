import os
from skimage.feature import hog
import numpy as np
import cv2
import pickle

#parameter for HOG features
pixpcell=(8,8)


print('Visualizing sample HOG features')
image=cv2.imread('C:\Users\liuyanghe\Desktop\TestImages\test-0.png',0)
sample,viz=hog(image, orientations=9, pixels_per_cell=pixpcell, block_norm='L2', cells_per_block=(2, 2), visualise=True)
viz=cv2.resize(viz,(459,300))
image=cv2.resize(image,(459,300))
cv2.imshow('image',image)
cv2.imshow('features',viz)
cv2.waitKey(0)
cv2.destroyAllWindows()