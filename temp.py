from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image

img = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'
img = Image.open(img)

mtcnn = MTCNN(image_size = 160, keep_all=True)
img_cropped, prob = mtcnn(img, return_prob = True)

img = img_cropped*255
img = (img.numpy())

for image_num in range(len(img)):
    image = img[image_num]

    img_new = image.astype(np.uint8)
    img_cropped_transposed = np.transpose(img_new, (1, 2, 0))
    image = Image.fromarray(img_cropped_transposed)


    image.show()
