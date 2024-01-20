from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import os

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

if __name__ == '__main__':
    img_path = 'temp_images/face_temp.jpg'
    _, temp = detect_face(img_path=img_path)

def is_inside(box1, box2):

    x1_inside = box2[0] <= box1[0] <= box1[2] <= box2[2]
    y1_inside = box2[1] <= box1[1] <= box1[3] <= box2[3]

    return x1_inside and y1_inside

