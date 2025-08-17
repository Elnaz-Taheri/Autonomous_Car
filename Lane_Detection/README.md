## Overview

This project implements a simulated autonomous car capable of detecting lane lines (yellow and white) and following them in real-time using computer vision. The system calculates the steering angle based on lane positions and controls the car in a simulator.

## Features

- Lane detection for yellow & white lines.
- Real-time steering angle calculation.
- Bird's-eye view perspective transform for accurate line detection.
- Adjustable car speed.
- Visualization of lane lines, lane center, steering angle, and speed.

## Requirements
- Python 3.7+ 

Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```

## Usage

### Connect to the simulator:
```
car = avisengine.Car()
if not car.connect("127.0.0.3", 25004):
    print("Connection failed")
    exit()
```


### Exit & stop car:

Press ESC to break the loop:
```
car.stop()
cv.destroyAllWindows()
```