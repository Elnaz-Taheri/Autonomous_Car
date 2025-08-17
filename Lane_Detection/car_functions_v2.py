import cv2 as cv
import numpy as np
from math import hypot

# Input image size (coming from the simulator)
W, H = 512, 512

# Car center in the image
CAR_CENTER = (260, 400)

# Perspective transform points (Bird-Eye View)
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (W - 30, H - 120)
bottom_left = (60, H - 120)

# Minimum contour area and maximum approximation
CONTOUR_MIN_SIZE = 250
APPROX_MAX_SIZE = 16

# Road line colors (HSV ranges)
LOWER_YELLOW = np.array([18, 75, 186])
UPPER_YELLOW = np.array([43, 255, 255])
LOWER_WHITE = np.array([0, 0, 184])
UPPER_WHITE = np.array([179, 44, 255])


def warp_frame(frame):
    """Convert main view to bird-eye view"""
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    return cv.warpPerspective(frame, matrix, (W, H))


def create_mask(frame, low, up):
    """Create a color mask (HSV)"""
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(img_hsv, low, up)


def find_line(frame, mask):
    """Find the line center from contours"""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    line_center_x = None
    line_img = np.zeros_like(frame)

    if len(contours) > 0:
        # Find the contour closest to the car
        candidates = []
        for c in contours:
            if cv.contourArea(c) > CONTOUR_MIN_SIZE:
                epsilon = 0.01 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                if len(approx) < APPROX_MAX_SIZE:
                    M = cv.moments(c)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        dist = hypot(cx - CAR_CENTER[0], cy - CAR_CENTER[1])
                        candidates.append((c, (cx, cy), dist))
        if candidates:
            candidates.sort(key=lambda x: x[2])
            best = candidates[0]
            line_center_x = best[1][0]
            cv.drawContours(line_img, [best[0]], -1, (0, 255, 0), -1)

    return line_img, line_center_x


def translate(value, leftMin, leftMax, rightMin, rightMax):
    """Map a pixel range to a steering angle range"""
    if leftMax == leftMin:
        return rightMin
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


def calc_steering(frame, prev_avg=None, debug=False):
    """Calculate the lane center (yellow + white lines)"""
    warped_frame = warp_frame(frame)

    yellow_mask = create_mask(warped_frame, LOWER_YELLOW, UPPER_YELLOW)
    white_mask = create_mask(warped_frame, LOWER_WHITE, UPPER_WHITE)

    yellow_img, yellow_center_x = find_line(warped_frame, yellow_mask)
    white_img, white_center_x = find_line(warped_frame, white_mask)

    # Decide which avg to use
    if yellow_center_x is not None and white_center_x is not None:
        avg = (yellow_center_x + white_center_x) // 2
    elif yellow_center_x is not None:
        avg = yellow_center_x
    elif white_center_x is not None:
        avg = white_center_x
    else:
        avg = prev_avg if prev_avg is not None else 0

    # Combine results
    result = cv.addWeighted(yellow_img, 0.5, white_img, 0.5, 0)

    if debug:
        cv.imshow("warped", warped_frame)
        cv.imshow("yellow mask", yellow_mask)
        cv.imshow("white mask", white_mask)

    return result, avg
