import numpy as np
import cv2 as cv

def greyScale(image):
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return imgray


def getTh(image):
    _, thresh = cv.threshold(image, 127, 255, 0)
    return thresh


def main():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)

    # Step 1:
    gr_image = greyScale(image)

    # Step 2:
    th_image = getTh(gr_image)

    # Step 3:

    # CHAIN_APPROX_NONE
    contours,_ = cv.findContours(th_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("CHAIN_APPROX_NONE \n",contours)


    # CHAIN_APPROX_NONE
    contours,_ = cv.findContours(th_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("\n\nCHAIN_APPROX_SIMPLE \n",contours)

    cv.waitKey(0)
    

main()




