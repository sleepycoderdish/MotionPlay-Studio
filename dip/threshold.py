import cv2
import numpy as np
def functionn(img, thresh, maxval, minval):
    newimg = np.where(img >= thresh, maxval, minval).astype(np.uint8)
    return newimg 
img = cv2.imread("gradient.jpg",0)
_,newimg = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
_,newimg2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)



# updated = functionn(img, 155, 255, 150)


cv2.imshow("img", img)
cv2.imshow("inverse", newimg)  
cv2.imshow("normal", newimg2)
# cv2.imshow("function_image", updated) 
cv2.waitKey(0)