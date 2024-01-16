from .utils import calc_dist, capture_images, generate_embedding, detect_face, display_img
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def new_user(save_path, embeddings):

    name_of_person = input('Enter the name of the person : ').lower()

    os.makedirs(os.path.join(save_path, name_of_person), exist_ok= True)

    face_path = os.path.join(save_path, name_of_person)
     
    name_of_embedding = os.path.join(face_path, name_of_person + '.npy')

    np.save(name_of_embedding, embeddings)

def check_user(all_img_path, img):

    _, curr_embedding = detect_face(img)
    if curr_embedding is not None:

        if len(os.listdir(all_img_path)) == 0:

            print('no existing user detected, adding new user....')

            return 'not found', curr_embedding
        
        else:

            for name in os.listdir(all_img_path):

                print('checking embedding with : ', name)

                name_dir = os.path.join(all_img_path, name)

                saved_embedding = np.load(os.path.join(name_dir, f'{name}.npy'))

                if calc_dist(curr_embedding, saved_embedding) < 1.0 : 
                    #print('distance is : ', calc_dist(curr_embedding, saved_embedding))
                    print("user exists, name : ", name)
                    return name, curr_embedding

            print('user does not exists, adding new user ...')
            return 'not found', curr_embedding
    else:

        print('Face not detected properly')
        return None

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("All files", "*.*")] )
    
    return file_path

def get_image(user_input):

    if user_input.lower() == "webcam":
        
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Capture")

        while True:

            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break

            cv2.imshow("Capture", frame)

            k = cv2.waitKey(1)
            if k % 256 == 32:
                cv2.imwrite('test.jpg', frame)
                cv2.destroyAllWindows()
                break

        return 'test.jpg'
           
    elif user_input.lower() == "image":
        # Ask the user to provide the image file path
        image_path = open_file_dialog()

        # Read the image
        img = cv2.imread(image_path)

        if img is not None:
            # Display the image
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return img
        else:
            print("Error: Unable to read the image.")

    else:
        print("Invalid option. Please enter 'webcam' or 'image'.")


if __name__ == '__main__':

    all_img_path = r'/home/harsh/AI-Projects/person-detection/images'
    curr_img_path = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'
    all_img_path = r'/home/harsh/AI-Projects/person-detection/images'
    check_user(all_img_path=all_img_path)
    breakpoint()