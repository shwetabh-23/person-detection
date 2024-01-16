from temp3 import get_boxes
from temp2 import get_faces, extract_faces
import cv2 
import PIL
from PIL import Image
import numpy as np
import os
from is_inside_box import is_inside
from facerecgnizer import check_user, new_user
import sys

def display_img_window(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    while True:
        cv2.namedWindow('Complete Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Complete Image', 900, 900)
        cv2.imshow('Complete Image', image_rgb)
        k = cv2.waitKey(0)
        if k % 256 == 32:
            cv2.destroyAllWindows()
            break


all_test_img_path = r'/home/harsh/AI-Projects/person-detection/test_images'

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
color = (255, 0, 0)
thickness = 4

for image in os.listdir(all_test_img_path):
    img = PIL.Image.open(os.path.join(all_test_img_path, image))
    img_numpy = np.array(img)
    x1, y1, x2, y2 = 0, 0, img_numpy.shape[1], img_numpy[0]
    display_img = img_numpy.copy()
    

    person_coords = get_boxes(display_img, display= False)
    face_coords = get_faces(img, display= False)

    for person in person_coords:
        for face in face_coords:
            temp_img = img_numpy.copy()

            if is_inside(face, person):

                x1_p, y1_p, x2_p, y2_p = person[0], person[1], person[2], person[3]
                x1_f, y1_f, x2_f, y2_f = face[0], face[1], face[2], face[3]

                face = img_numpy[y1_f  - 10 :y2_f + 10 , x1_f - 10 : x2_f + 10]
               
                face = PIL.Image.fromarray(face)

                face = extract_faces(face)
                cv2.rectangle(temp_img, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 4)
                display_img_window(temp_img)
                #Image.fromarray(temp_img).show()

                #check = input('Please verify the image to be detected, press c to continue or q to quit : ')

                all_img_path = r'images'
                name, curr_embedding = check_user(all_img_path=all_img_path, img=face)

                if name == 'not found':

                    user_choice = input('Do you want to name this person, say Yes or No : ').lower()

                    if user_choice == 'yes':

                        new_user(all_img_path, curr_embedding)

                        print('new user added')
                        
                    elif user_choice == 'no':

                        name = 'unknown'

                        continue

                    else:
                        print('Invalid entry, taking no as an answer')
                        continue
                
                elif name is None:
                    continue
                else:
                    cv2.rectangle(display_img, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 4)
                    cv2.rectangle(display_img, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 4)
                    org = [x1_p, y1_p]
                    cv2.putText(display_img, name, org, font, fontscale, color, thickness)
    display_img_window(display_img)