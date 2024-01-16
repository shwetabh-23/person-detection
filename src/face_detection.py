from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image

mtcnn = MTCNN(image_size = 160)

def get_faces(img, window = 'Image', display = False):
    boxes, prob = mtcnn.detect(img)
    coords = []
    img = np.array(img)
    counter = 0
    for box in boxes:
        curr_prob = prob[counter]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        if curr_prob > 0.9:
            coords.append([x1, y1, x2, y2])
        counter += 1

    if display == True:
        cv2.imshow(window, img)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k%256 == 32:
                cv2.destroyAllWindows()
                break
    return coords

def extract_faces(img):
    
    # Detect faces
    face = mtcnn(img)
    
    return face

if __name__ == '__main__':

    img = r'temp_images/face_temp.jpg'
    img = Image.open(img)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 900, 900)

    get_faces(img=img, display= True)
