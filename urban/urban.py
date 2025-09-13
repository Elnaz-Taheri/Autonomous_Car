import cv2 as cv
import avisengine
from config import *   # includes speed
import numpy as np
import onnxruntime as ort
from time import time
from math import hypot

# ------------------ settings ------------------
classes = ['Proceed Forward', 'Proceed Left', 'Proceed Right', 'Stop','traffic light']
NUM_CLASSES = len(classes)
CONF_THRES = 0.5
NMS_THRES = 0.6
# ------------------ settings ------------------
W, H = 512, 512
CAR_CENTER = (260, 400)
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (W - 30, H - 120)
bottom_left = (60, H - 120)
CONTOUR_MIN_SIZE = 250
APPROX_MAX_SIZE = 16
LOWER = np.array([0, 11, 148])
UPPER = np.array([41, 19, 255])


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
    """Calculate lane center based on white line with memory of previous frame."""
    warped_frame = warp_frame(frame)
    white_mask = create_mask(warped_frame, LOWER, UPPER)
    white_img, white_center_x = find_line(warped_frame, white_mask)

    # Use previous avg if white line is not detected
    if white_center_x is not None:
        target_x = white_center_x  
    else:
        target_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    # Proportional control for smooth steering
    current_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    if debug:
        cv.imshow("warped", warped_frame)
        cv.imshow("white mask", white_mask)

    return white_img, target_x 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def iou(b1, b2):
    x1,y1,x2,y2 = b1
    X1,Y1,X2,Y2 = b2
    inter = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))
    area1, area2 = (x2-x1)*(y2-y1), (X2-X1)*(Y2-Y1)
    return inter / (area1 + area2 - inter + 1e-6)

# ------------------ Connect to car ------------------
car = avisengine.Car()
if not car.connect("127.0.0.3", 25004):
    print("❌ Connection failed")
    exit()


# ------------------ Load ONNX model ------------------
model = ort.InferenceSession("best_detect.onnx")

traffic_light_detected = False
start_time = 0
center_line_x = 0

while True:
    car.getData()
    frame = car.getImage()
    if frame is None:
        continue

    # ------------------ Lane Detection ------------------
    lane_img, center_line_x = calc_steering(frame, prev_avg=center_line_x)   


    orig_h, orig_w = frame.shape[:2]

    # --------- Preprocess input ---------
    inp = cv.resize(frame, (384, 384)).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None]   # (1, 3, 384, 384)

    # --------- Run model ---------
    outputs = model.run(None, {"images": inp})
    out0 = outputs[0]          # (1, 9, 3024)
    out0 = out0.squeeze(0)     # (9, 3024)
    out0 = out0.transpose(1,0) # (3024, 9)

    boxes = out0[:, :4]        # (3024, 4)
    scores = out0[:, 4:]       # (3024, 5)

    detections = []
    for i, box in enumerate(boxes):
        probs = scores[i]
        cid = probs.argmax()
        conf = probs[cid]
        if conf < CONF_THRES:
            continue
    
        xc, yc, w, h = box
        x1 = (xc - w/2) / 384 * orig_w
        y1 = (yc - h/2) / 384 * orig_h
        x2 = (xc + w/2) / 384 * orig_w
        y2 = (yc + h/2) / 384 * orig_h

        detections.append([x1, y1, x2, y2, classes[cid], conf])


    # --------- NMS ---------
    detections.sort(key=lambda x: x[5], reverse=True)
    final_dets = []
    while detections:
        best = detections.pop(0)
        final_dets.append(best)
        detections = [d for d in detections if iou(d[:4], best[:4]) < NMS_THRES]

    # --------- Control car ---------
    avg=center_line_x //2
    traffic_light = False
    Proceed_Forward = False
    Proceed_Left = False
    Proceed_Right = False
    Stop = False
 
    if final_dets:  
        l_2 = final_dets[0][4]
        s_2= final_dets[0][5]
        
        if l_2 == 'traffic light' and s_2>0.8 and final_dets[0][2]>190 and final_dets[0][2]<350:
            traffic_light = True
            print(l_2,s_2,final_dets[0][2])  

        elif  l_2 == 'Proceed Forward' and s_2>0.8 :
            Proceed_Forward = True
            print(l_2,s_2) 
        elif  l_2 == 'Proceed Left' and s_2>0.8 :
            Proceed_Left = True
            print(l_2,s_2 )
        elif  l_2 == 'Proceed Right' and s_2>0.8 :
            Proceed_Right = True
            print(l_2,s_2)
        elif  l_2 == 'Stop' and s_2>0.8 :
           Stop = True
           print(l_2,s_2)
        else:
           traffic_light = False    
           Proceed_Forward = False
           Proceed_Left = False
           Proceed_Right = False
           Stop = False


    if   traffic_light :
        car.setSpeed(0)
        start_time = time()
        
    else:
        steering = translate(avg,90,170,-15,15)
        car.setSteering(int(steering))
        
      
    if traffic_light and ( time() - start_time < 20) : 
            traffic_light=False
            if Proceed_Right:
                steering = translate(center_line_x ,0,50, 0 ,90)
                car.setSteering(int(steering))
            elif Proceed_Left:
                steering = translate(center_line_x ,0, 50 , -90 ,0)
                car.setSteering(int(steering))
            elif Proceed_Forward:
                steering = translate(center_line_x,90,170,-15,15)
                car.setSteering(int(steering))
            elif Stop:
                car.setSpeed(0)
        
           
    else:
        car.setSpeed(10)
        traffic_light = False
    
            
      


    # --------- Draw results ---------
    show_frame = frame.copy()
    for x1, y1, x2, y2, label, conf in final_dets:
        cv.rectangle(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv.putText(show_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv.putText(show_frame, f"Speed: {car.getSpeed()}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv.imshow("Object Detection", show_frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break

car.stop()
cv.destroyAllWindows()
