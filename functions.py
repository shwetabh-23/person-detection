import cv2 
import PIL
import numpy as np

from src import get_boxes, get_faces, extract_faces, is_inside, check_user, new_user, detect_face

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
thickness = 2

def get_name_image(img, image_name):
        
        img_numpy = np.array(img)
        display_img = img_numpy.copy()
        person_coords = get_boxes(display_img, display= False)
        face_coords = get_faces(img, display= False)

        for person in person_coords:
            for face in face_coords:
                temp_img = img_numpy.copy()

                if is_inside(face, person):

                    x1_p, y1_p, x2_p, y2_p = person[0], person[1], person[2], person[3]
                    x1_f, y1_f, x2_f, y2_f = face[0], face[1], face[2], face[3]

                    face = img_numpy[y1_f  - 20 :y2_f + 20 , x1_f - 20 : x2_f + 20]
                    face = PIL.Image.fromarray(face)

                    face = extract_faces(face)
                    cv2.rectangle(temp_img, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 4)
                    check = check_user(img=face)
                    if check == True:
                    
                        #cv2.rectangle(display_img, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 4)
                        #cv2.rectangle(display_img, (x1_p, y1_p), (x2_p, y2_p), (0, 0, 255), 4)
                        #org = [x1_p, y1_p + 50]
                        #cv2.putText(display_img, name, org, font, fontscale, color, thickness)

                        save_img = display_img[y1_p : y2_p, x1_p : x2_p]
                        try:
                            image_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f'static/{image_name}', image_bgr)
                            return image_bgr
                        except:

                            continue
import os

def add_new_user(image, name):
    image = np.array(image)
    face = extract_faces(image)
    img, embedding = detect_face(img=face)
    np.save(os.path.join(r'/home/harsh/AI-Projects/person-detection/generated_embeddings', f'{name}.npy'), embedding)

if __name__ == '__main__':
    img = r'/home/harsh/AI-Projects/person-detection/test_images/IMG20230511020948.jpg'
    img = cv2.imread(img)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    get_name_image(image_rgb)    
