from .utils import calc_dist, capture_images, generate_embedding, detect_face, display_img
import os
import numpy as np
import cv2

def new_user(save_path):

    name_of_person = capture_images(save_path=save_path)

    os.makedirs(os.path.join(save_path, name_of_person), exist_ok= True)
    face_path = os.path.join(save_path, name_of_person)
    #breakpoint()
    embeddings = generate_embedding(save_path=save_path, face_path=face_path)

    name_of_embedding = os.path.join(face_path, name_of_person + '.npy')

    np.save(name_of_embedding, embeddings)

def check_user(all_img_path):

    cam = cv2.VideoCapture(-1, cv2.CAP_V4L2)
    cv2.namedWindow("Capture")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Capture", frame)

        k = cv2.waitKey(1)
       
        if k%256 == 32:
            try : 
                cv2.imwrite(os.curdir + 'test.png', frame)
                img_path = os.curdir + 'test.png'
                _, curr_embed = detect_face(img_path=img_path)
                distance = {}
                for name in os.listdir(all_img_path):
                    curr_dir = os.path.join(all_img_path, name)
                    check_embed = np.load(os.path.join(curr_dir, name+'.npy'))
                    dist = calc_dist(curr_embed, check_embed)
                    distance[dist] = name
                min_dist = min(distance.keys())
                #breakpoint()
                if min_dist < 0.7:
                    os.remove(img_path)
                    cam.release()
                    cv2.destroyAllWindows()
                    return True, distance[min_dist]
                
                else:
                    os.remove(img_path)
                    cam.release()
                    cv2.destroyAllWindows()
                    return False, name
            except : 
                return False, 'none'

    

    

if __name__ == '__main__':
    check_user()