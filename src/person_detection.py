import cv2
from ultralytics import YOLO
import yaml
import math

model = YOLO(r'yolo-Weights/yolov8n.pt')

img_path = r'test_images/74469533_2443137869348818_8858146082635382784_n.jpg'
img = cv2.imread(img_path)

classes_path = r'/home/harsh/AI-Projects/person-detection/classes.yaml'

with open(classes_path, 'r') as classes:
    classes = yaml.load(classes, Loader= yaml.FullLoader)

def get_boxes(img, display = False):

    results = model(img)

    coords = []

    for r in results:

        boxes = r.boxes
        
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            coords.append([x1, y1, x2, y2])
            
        if display == True:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image', 1500, 1500)
            while True:
                cv2.imshow('Image', img)
                k = cv2.waitKey(0)
                if k % 256 == 32:
                    #cv2.destroyAllWindows()
                    break
        return coords

#get_boxes(img=img, display= True)