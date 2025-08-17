import cv2 as cv
import avisengine
import car_functions_v2 as cf
from config import *   # includes speed

# Create the car object and connect to the simulator
car = avisengine.Car()
if not car.connect("127.0.0.3", 25004):
    print("❌ Connection failed")
    exit()

# Initial value for avg
avg = 0
while True:
    car.getData()
    car.setSpeed(speed)  # speed value is defined in config (e.g., 50)

    frame = car.getImage()
    if frame is None:
        continue

    # Calculate avg from the lane lines
    lane_img, avg = cf.calc_steering(frame, prev_avg=avg, debug=False)

    # Convert avg in pixels to steering angle
    steering = cf.translate(avg, 220, 300, -15, 15)
    car.setSteering(int(steering))

    # For display
    show_frame = frame.copy()
    cv.circle(show_frame, (avg, 300), 5, (0, 255, 0), cv.FILLED)
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
