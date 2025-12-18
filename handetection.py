import cv2
import numpy as np

def empty(a):
    pass

def create_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 500, 300)
    cv2.createTrackbar('HueMin', 'Trackbars', 0, 179, empty)
    cv2.createTrackbar('HueMax', 'Trackbars', 179, 179, empty)
    cv2.createTrackbar('SatMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('SatMax', 'Trackbars', 255, 255, empty)
    cv2.createTrackbar('ValMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('ValMax', 'Trackbars', 60, 255, empty)

def create_mask(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos('HueMin', 'Trackbars')
    hue_max = cv2.getTrackbarPos('HueMax', 'Trackbars')
    sat_min = cv2.getTrackbarPos('SatMin', 'Trackbars')
    sat_max = cv2.getTrackbarPos('SatMax', 'Trackbars')
    val_min = cv2.getTrackbarPos('ValMin', 'Trackbars')
    val_max = cv2.getTrackbarPos('ValMax', 'Trackbars')
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    return mask

def find_and_draw_centroid(image,contours):
    # TODO COMPLETE THIS FUNCTION SO THAT IT RETURNS AN IMAGE WITH CENTROIDS DRAWN
    for contour in contours:
        M = cv2.moments(contour)
        if(M['m00'] != 0):
            Mx = int(M['m10']/M['m00'])
            My = int(M['m01']/M['m00'])

            cv2.circle(image, (Mx,My), 2, (0,0,255), -1)        
    return image

def main():
    vid = cv2.VideoCapture(0)
    create_trackbars()

    while True:
        ret, image = vid.read()
        image_thres = create_mask(image)
        cv2.imshow('Thres',image_thres)

        contours,_  = cv2.findContours(image_thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image,contours,-1,(0,255,0),2)

        image = find_and_draw_centroid(image,contours)
        cv2.imshow("contours",image)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()




main()