import cv2
import numpy as np


# Mozliwe, ze trzeba doinstalowac pakiet opencv-contrib-python w konkretnej wersji:
# [root@laptop]# pip3.6 uninstall opencv-contrib-python
# [root@laptop]# pip3.6 install opencv-contrib-python==3.3.0.10


img = cv2.imread('IM.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray,None)
cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('IM_sift_keypoints.png',img)
