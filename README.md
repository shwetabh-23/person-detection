# Face Recognition and Cropping Tool

## Overview

This project is a Python-based tool for face recognition and cropping within group photos. It allows users to take a photo with multiple people, draws bounding boxes around each person and their face, and returns the person's name if the code already has the face embeddings for it. If the face embeddings are not available, the tool prompts the user to create the face embedding and adds it to the directory. Finally, the tool returns the cropped images of each person along with their names.

## Features

1. Face recognition within group photos.
2. Bounding box drawing around each person and their face.
3. Identification of known faces based on pre-existing face embeddings.
4. On-the-fly creation and addition of face embeddings for unknown faces.
5. Cropped images of each person returned for further manipulation.

## Models Used

This tool leverages the following models for its functionalities:

1. FaceNet: Used for face recognition and creating face embeddings.
2. MTCNN (Multi-task Cascaded Convolutional Networks): Utilized for detecting faces and creating bounding boxes.
3. YOLOv8 (You Only Look One): Applied for person detection within the group photo.

## Usage

1. Take a group photo containing multiple people.
2. Run the script using python main.py.
3. The script will draw bounding boxes around each person and their face.
4. If a known face is recognized, the person's name will be displayed.
5. If the face is unknown, the user will be prompted to create a face embedding and add it to the directory.
6. Cropped images of each person along with their names will be saved for further use.

## Reproduce locally 

1. run the command setup_env.sh

bash ./setup_env.sh

2. run the command run_app.sh

bash ./run_app.sh

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.