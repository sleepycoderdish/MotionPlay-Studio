import numpy as np
import cv2 as cv

def greyScale(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return imgray


def getTh(image):
    _, thresh = cv.threshold(image, 127, 255, 0)
    return thresh


def main():
    image = cv.imread('assets/hierarchy.png')

    # Step 1:
    gr_image = greyScale(image)

    # Step 2:
    th_image = getTh(gr_image)

    # Step 3:

    # RETR_LIST
    contours,hierarchy = cv.findContours(th_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    print("RETR_LIST \n",hierarchy)

    # RETR_EXTERNAL
    contours,hierarchy = cv.findContours(th_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print("\n RETR_EXTERNAL \n",hierarchy)

    # RETR_TREE
    contours,hierarchy = cv.findContours(th_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("\n RETR_TREE \n",hierarchy)



    cv.drawContours(image,contours,-1,(0,255,0),1)
    cv.imshow('image',image)
    cv.waitKey(0)
    

main()




