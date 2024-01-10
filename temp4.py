from temp3 import get_boxes
from temp2 import get_faces
import cv2 
import PIL
import numpy as np

img_path = r'/home/harsh/AI-Projects/person-detection/IMG20231203223631.jpg'

img = PIL.Image.open(img_path)
img_numpy = np.array(img)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 900, 900)

img = get_boxes(img_numpy, display= True)
get_faces(img, display= True)