from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from temp2 import get_faces

# Create MTCNN face detection model
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False)

# Load image


# Check if faces are detected
def display_faces(faces):
    if faces is not None:
        for i in range(len(faces)):
            face_np = np.array(faces[i].permute(1, 2, 0), dtype=np.uint8)
            plt.imshow(face_np)
            plt.title('Detected Face')

            # Show the plot
            plt.show()
    else:
        print("No faces detected.")

def get_bb_and_face(img_path):

    img = Image.open(img_path)

    img = np.array(img)

    boxes = get_faces(img=img)

    for face in boxes :

        x1_f, y1_f, x2_f, y2_f = face[0], face[1], face[2], face[3]

        curr_face = img[y1_f - 100 : y2_f + 100 , x1_f - 100 : x2_f + 100]

        curr_face = Image.fromarray(curr_face)

        curr_face.show()

        face = extract_faces(curr_face)

        display_faces(faces=face)

if __name__ == '__main__':

    img_path = r'/home/harsh/AI-Projects/person-detection/test_images/IMG20230909184009.jpg'

    get_bb_and_face(img_path=img_path)



