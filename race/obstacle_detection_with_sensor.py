import cv2 as cv
import avisengine
from config import *   # includes speed
from time import time
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
LOWER_YELLOW = np.array([22, 102, 122])
UPPER_YELLOW = np.array([30, 255, 255])

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
    """Calculate lane center based on yellow line with memory of previous frame."""
    warped_frame = warp_frame(frame)
    yellow_mask = create_mask(warped_frame, LOWER_YELLOW, UPPER_YELLOW)
    yellow_img, yellow_center_x = find_line(warped_frame, yellow_mask)

    # Use previous avg if yellow line is not detected
    if yellow_center_x is not None:
        target_x = yellow_center_x  
    else:
        target_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    # Proportional control for smooth steering
    current_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    if debug:
        cv.imshow("warped", warped_frame)
        cv.imshow("yellow mask", yellow_mask)

    return yellow_img, target_x 

# Create the car object and connect to the simulator
car = avisengine.Car()
if not car.connect("127.0.0.3", 25004):
    print("❌ Connection failed")
    exit()

# Initial value for avg
center_line_x= 0

obs = False

car.setSensorAngle(20)


while True:
    car.getData()
    car.setSpeed(speed)  # speed value is defined in config (e.g., 50)
    sensors = car.getSensors()

    frame = car.getImage()

    if frame is None:
        continue
    # Calculate avg from the lane lines
    lane_img, center_line_x = calc_steering(frame, prev_avg=center_line_x, debug=False)


    if abs(car.steering_value)<10:
        car.setSpeed(speed)
    else:
        car.setSpeed(7)

    if sensors[2] > 1100:
        # Map to steering
        # Convert avg in pixels to steering angle
        steering = translate(center_line_x ,0, 171, -30, 30)
        car.setSteering(int(steering))
    else:
        if obs == False:
            obs = True
            start_time = time()

    if obs and ( time() - start_time < 10) :
            steering = translate(center_line_x ,340, 400 , -80, 80)
            car.setSteering(int(steering))
            if car.get_Speed > 10:
               car.setSpeed(7)
    else:
        obs = False
    # For display
    show_frame = frame.copy()
    cv.circle(show_frame, (center_line_x, 300), 5, (0, 255, 0), cv.FILLED)
    cv.putText(show_frame, f"Steering: {int(steering)}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(show_frame, f"Speed: {car.getSpeed()}",
               (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv.imshow("frame", show_frame)
    cv.imshow("lane detection", lane_img)

    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break

car.stop()
cv.destroyAllWindows()