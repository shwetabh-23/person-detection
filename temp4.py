from temp3 import get_boxes
from temp2 import get_faces
import cv2 
import PIL
import numpy as np
import os
from is_inside_box import is_inside

all_img_path = r'/home/harsh/AI-Projects/person-detection/images'

for image in os.listdir(all_img_path):
    img = PIL.Image.open(os.path.join(all_img_path, image))
    img_numpy = np.array(img)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 900, 900)

    person_coords = get_boxes(img_numpy, display= False)
    face_coords = get_faces(img, display= False)

    for person in person_coords:
        for face in face_coords:
            if is_inside(face, person):

                x1_p, y1_p, x2_p, y2_p = person[0], person[1], person[2], person[3]
                x1_f, y1_f, x2_f, y2_f = face[0], face[1], face[2], face[3]
                
                cv2.rectangle(img_numpy, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 4)
                cv2.rectangle(img_numpy, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 4)

                cv2.imwrite(f'{image}_edited.jpg,', img_numpy)
    while True:
        cv2.imshow('Image', img_numpy)
        k = cv2.waitKey(0)
        if k % 256 == 32:
            cv2.destroyAllWindows()
            break