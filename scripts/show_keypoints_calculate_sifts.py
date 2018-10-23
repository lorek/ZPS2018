import cv2
import numpy as np


# Mozliwe, ze trzeba doinstalowac pakiet opencv-contrib-python w konkretnej wersji:
# [root@laptop]# pip3.6 uninstall opencv-contrib-python
# [root@laptop]# pip3.6 install opencv-contrib-python==3.3.0.10


img = cv2.imread('IM.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

#tylko wykrycie keypointow:
kp = sift.detect(gray,None)
cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('IM_sift_keypoints.png',img)


#wykrycie keypointow i wyliczenie tam siftow:

kp, sifts = sift.detectAndCompute(gray,None)


print("Mamy ",sifts.shape[0], " punktow ", sifts.shape[1],"-wymiarowych")
print("\n")
print(type(sifts)) # jak widac, jest to maciec NumPy

print(sifts)
