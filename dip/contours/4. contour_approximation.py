import numpy as np
import cv2 as cv

def greyScale(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return imgray


def getTh(image):
    _, thresh = cv.threshold(image, 127, 255, 0)
    return thresh


def main():
    image = cv.imread('assets/approx.png')
    image_cpy = image.copy()

    # Step 1:
    gr_image = greyScale(image)

    # Step 2:
    th_image = getTh(gr_image)

    # Step 3:
    contours,_ = cv.findContours(th_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(image,contours,-1,(0,255,0),2)
    cv.imshow("image contours",image)

    
    epsilon = 0.1*cv.arcLength(contours[0],True)
    approx = cv.approxPolyDP(contours[0],epsilon,True)
    cv.drawContours(image_cpy,[approx],0,(0,255,0),2)
    cv.imshow("image approx",image_cpy)


    cv.waitKey(0)
    

main()




