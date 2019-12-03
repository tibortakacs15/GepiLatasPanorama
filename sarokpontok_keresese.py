import cv2
import numpy as np


# kepek beolvasasa
I = cv2.imread("building1.jpg")  
I2 = cv2.imread("building2.jpg") 
#kepek syurke arnzalatossa atalakitasa
Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I2g = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

#kulcspontok(sarokpontok) meghatarozasa
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(I, None)
kp2, des2 = sift.detectAndCompute(I2g, None)
#2 kep kulcspontjainak osszekapcsolasa
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
I3 = cv2.drawMatchesKnn(I,kp1,I2,kp2,good,None,flags=2)

cv2.imshow("Kep", I3)
cv2.waitKey(0)#var egy billentyuzet lenyomasara
cv2.destroyAllWindows()# megsemmisiti
