from argparse import ArgumentParser

import cv2
import numpy as np


parser = ArgumentParser()
parser.add_argument('--scale', type=float, default=2.,
                    help='Scale factor to be used')
args = parser.parse_args()


def temp(temp_value): ...


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],(0,0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray.shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


h_min = 18
h_max = 43
s_min = 75
s_max = 255
v_min = 186
v_max = 255

img = cv2.imread("race.png")
#print(f'{img.shape = }')

# color_name = args.name

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", h_min, 179, temp)
cv2.createTrackbar("Hue Max", "TrackBars", h_max, 179, temp)
cv2.createTrackbar("Sat Min", "TrackBars", s_min, 255, temp)
cv2.createTrackbar("Sat Max", "TrackBars", s_max, 255, temp)
cv2.createTrackbar("Val Min", "TrackBars", v_min, 255, temp)
cv2.createTrackbar("Val Max", "TrackBars", v_max, 255, temp)

while True:
    img = cv2.resize(cv2.imread("race.png"), (0, 0), fx=.5, fy=.5)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(f"h_min: {h_min}, h_max: {h_max}, s_min: {s_min}, s_max: {s_max}, v_min: {v_min}, v_max: {v_max}")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgStack = stackImages(args.scale, ([img, imgHSV], [mask, imgResult]))
    cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
