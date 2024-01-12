from utils import calc_dist, capture_images, generate_embedding, detect_face, display_img
import os
import numpy as np
import cv2

def new_user(save_path, image, name_of_person):

    os.makedirs(os.path.join(save_path, name_of_person), exist_ok= True)

    face_path = os.path.join(save_path, name_of_person)
     
    embeddings = generate_embedding(save_path=save_path, face_path=face_path)

    name_of_embedding = os.path.join(face_path, name_of_person + '.npy')

    np.save(name_of_embedding, embeddings)

def check_user(all_img_path, curr_img_path):

    _, curr_embedding = detect_face(curr_img_path)

    if len(os.listdir(all_img_path)) == 0:

        print('no existing user detected')

        new_user(all_img_path, curr_img_path, 'shwetabh')

    else:

        for name in os.listdir(all_img_path):

            name_dir = os.path.join(all_img_path, name)

            saved_embedding = os.path.join(name_dir, f'{name}.npy')

            if calc_dist(curr_embedding, saved_embedding) < 0.7 : 

                print("user exists")

            else : 

                print('user does not exists')

                new_user(all_img_path, curr_img_path)


        
    
if __name__ == '__main__':

    all_img_path = r'/home/harsh/AI-Projects/person-detection/images'
    curr_img_path = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'

    check_user(all_img_path=all_img_path, curr_img_path=curr_img_path)