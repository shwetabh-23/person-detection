from ultralytics import YOLO
import cv2
import math




model = YOLO('yolo-Weights/yolov8n.pt')

classlist = {0 : 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stopsign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

classes = list(classlist.values())

def object_detection(display = True):

    cap = cv2.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 512)
    
    while True:

        success, img = cap.read()
        breakpoint()
        results = model(img, stream = True)
        detected_class = []
        for r in results:

            boxes = r.boxes
            
            for box in boxes:
                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                conf = math.ceil((box.conf[0] * 100)/100)
                #print("confidence : {}".format(conf))
                
                cls = int(box.cls[0])
                #print("class : {}".format(classes[cls]))
                
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontscale = 1
                color = (255, 0, 0)
                thickness = 2
                
                cv2.putText(img, classes[cls], org, font, fontscale, color, thickness)
                #print(classes[cls])
                detected_class.append(classes[cls])
            if display == True:
                cv2.imshow('Webcam', img)

        k = cv2.waitKey(1) & 0xFF
        if k%256 == 32:
            cap.release()
            cv2.destroyAllWindows()
            return list(set(detected_class))
        
if __name__ == '__main__':
    object_detection()