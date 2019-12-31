import cv2
import numpy as np

# kepek beolvasasa
I = cv2.imread("minjtemplom.jpg")
Ig = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
I2 = cv2.imread("btemplom.jpg")
I2g = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# kulcspontok keresése
kp1, des1 = sift.detectAndCompute(Ig,None)
kp2, des2 = sift.detectAndCompute(I2g,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# kulcspontok parositasanak szurese
good = []
for i,j in matches:
    if i.distance < 0.5*j.distance:
        good.append([i])
        matches = np.array(good)
#homografia matrix meghatarozasa
if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[i.queryIdx].pt for i in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[i.trainIdx].pt for i in matches[:,0] ]).reshape(-1,1,2)
    H, maszk = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    print("Nem talalhato eleg kozos kulcspont!")

Ihomografia = cv2.warpPerspective(I,H,(I.shape[1] + I2.shape[1], I.shape[0]+I2.shape[0]))
Ihomografia[0:I2.shape[0], 0:I2.shape[1]] = I2
#fekete szin levagasa
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
filename = 'mtemplomPK.jpg'
cv2.imwrite(filename, trim(Ihomografia))
cv2.imshow("Panorama kep", trim(Ihomografia))
cv2.imshow("kep1", I)
cv2.imshow("kep2", I2)
cv2.waitKey(0)#var egy billentyuzet lenyomasara
cv2.destroyAllWindows()# megsemmisiti



