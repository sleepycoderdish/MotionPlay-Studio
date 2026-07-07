import cv2
img = cv2.imread("boat.jpg",0)


# otsu code
_,otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, cv2.THRESH_BINARY)


cv2.imshow("otsu",otsu)

cv2.waitKey(0)