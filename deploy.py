import ultralytics
import cv2

model = ultralytics.YOLO('yolo26m.pt')

# Input: number = video input, filename of image, or path to folder of images
result = model("savanna.jpg")
"""
result is a list, with one entry per image 
Each entry is a Results object.  
Two useful things about a Results object:
    1. It has a plot() method, which outputs a version of the original image
        with all the detected boxes on it.
    2. It has a boxes attribute, which contains information about all the 
        boxes.  Notice that xywh coordinates are x,y of *center*, then w,h.
"""
print(result[0].boxes)
cv2.imshow("", result[0].plot())
cv2.waitKey()