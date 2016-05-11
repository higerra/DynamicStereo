import cv2
import numpy as np

img = cv2.imread('/home/yanhang/Documents/research/DynamicStereo/data/data_newyork2/images/image00000.jpg')
cv2.imshow('a', img)
k = cv2.waitKey(0) & 0XFF
cv2.destroyAllWindows()