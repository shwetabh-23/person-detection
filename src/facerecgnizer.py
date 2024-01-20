from .utils import calc_dist, detect_face
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

def check_user(img):

    _, curr_embedding = detect_face(img)
    if curr_embedding is not None:


        saved_embedding = np.load(r'/home/harsh/AI-Projects/person-detection/curr_embeddings.npy')

        if calc_dist(curr_embedding, saved_embedding) < 1.0 : 
            print('distance is : ', calc_dist(curr_embedding, saved_embedding))
            return True

    else:

        print('Face not detected properly')
        return None

if __name__ == '__main__':

    all_img_path = r'/home/harsh/AI-Projects/person-detection/images'
    curr_img_path = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'
    all_img_path = r'/home/harsh/AI-Projects/person-detection/images'
    check_user(all_img_path=all_img_path)
    breakpoint()