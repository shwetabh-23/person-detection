from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import os
import sys
import torch

mtcnn = MTCNN(image_size=160)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

def detect_face(img):

    try:

        tensor_img = img.unsqueeze(0)
       
        img_embedding = resnet(tensor_img)
        img_embedding = img_embedding.detach().numpy().flatten()

        print('face detected')

        return img, img_embedding
    except:

        print('Face cannot be detected, try again')
        return None, None

def display_img(img_cropped):

    img = img_cropped*255
    img = (img.numpy())
    print(img)
    img = img.astype(np.uint8)

    img_cropped_transposed = np.transpose(img, (1, 2, 0))

    image = Image.fromarray(img_cropped_transposed)

    image.show()

def calc_dist(embedding1, embedding2):

    try :
        result = np.sqrt(np.sum(np.square(embedding2 - embedding1)))
        print('distance : ', result)
        return result

    except :
        return float('inf')

def capture_images(save_path):
    
    cam = cv2.VideoCapture(0)

    name_of_person = input('Enter the name of the person in the frame : ')
    cv2.namedWindow("Capture")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Capture", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            while img_counter < 15:
                os.makedirs(os.path.join(save_path, name_of_person), exist_ok= True)
                save_path_n = os.path.join(save_path, name_of_person)
                last_check = 0
                if len(os.listdir(save_path_n)) > 0:
                    if os.path.exists(os.path.join(save_path_n, name_of_person + '.npy')):

                        last_check = int(os.listdir(save_path_n)[-2].split('-')[-1][1])

                img_name = os.path.join(save_path_n, ("{} - {}.png".format(name_of_person, img_counter + last_check)))
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
            cam.release()
            cv2.destroyAllWindows()
            break
    
    return name_of_person

def generate_embedding(save_path, face_path):
    
    faces = [os.path.join(face_path, i) for i in os.listdir(face_path)]
    embeddings = []
    for face in faces:
        _, embedding = detect_face(os.path.abspath(face))
        embeddings.append(embedding)
    avg_embeds = (np.mean(np.array(embeddings), axis = 0))
    return avg_embeds

if __name__ == '__main__':
    img_path = 'temp_images/face_temp.jpg'
    _, temp = detect_face(img_path=img_path)
