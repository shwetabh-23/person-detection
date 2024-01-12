import cv2
from ultralytics import YOLO
import yaml

model = YOLO(r'yolo-Weights/yolov8n.pt')

img_path = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'

img = cv2.imread(img_path)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 900, 900)

classes_path = r'/home/harsh/AI-Projects/person-detection/classes.yaml'

with open(classes_path, 'r') as classes:
    classes = yaml.load(classes, Loader= yaml.FullLoader)

def get_boxes(img, window = 'Image', display = False):

    results = model(img)

    coords = []

    for r in results:

        boxes = r.boxes
        
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            coords.append([x1, y1, x2, y2])
            cls = int(box.cls[0])
            
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 3
            color = (255, 0, 0)
            thickness = 5
            
            #cv2.putText(img, f'{x1}, {y1}, {x2}, {y2}', org, font, fontscale, color, thickness)

            cv2.imshow(window, img)
        if display == True:
            while True:
                k = cv2.waitKey(0)
                if k % 256 == 32:
                    #cv2.destroyAllWindows()
                    break
        return coords

