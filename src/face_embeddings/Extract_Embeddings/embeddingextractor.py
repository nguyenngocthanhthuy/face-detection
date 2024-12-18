import os
import numpy as np
from deepface import DeepFace
import cv2
from face_lib import face_lib


import cv2
import numpy as np


class CropFace:
    def __init__(self) -> None:
        self.FL = face_lib()

    def crop_face_img(self, input_frame):
        # Check if the input is a path and read the image if it is
        if isinstance(input_frame, str):
            frame = cv2.imread(input_frame)
        else:
            frame = input_frame

        frame_height, frame_width = frame.shape[:2]

        faces, boxes = self.FL.faces_locations(frame)
        if faces > 0:
            x, y, w, h = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

            x = max(0, x - 10)
            y = max(0, y - 50)
            w = min(frame_width, x + w + 15)
            h = min(frame_height, y + h + 75)

            return frame[y:h, x:w]

        return frame


class Embeddings:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_embeddings(self, img):
        vector = DeepFace.represent(
            img, model_name=self.model_name, enforce_detection=False
        )[0]["embedding"]

        return vector


class EmbeddingExtractor(CropFace, Embeddings):
    def __init__(self, dataset_path, model_name):
        CropFace.__init__(self)
        Embeddings.__init__(self, model_name)
        self.dataset_path = dataset_path

    def extract_embeddings(self):
        embeddings = []
        labels = []
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    image = cv2.imread(img_path)
                    img_crop = self.crop_face_img(image)
                    embedding = self.get_embeddings(img_crop)
                    embeddings.append(embedding)
                    labels.append(class_name)
        return np.array(embeddings), labels


# Usage

if __name__ == "__main__":
    dataset_path = r"C:\Users\nguye\Downloads\Doan\sp_project-main\sp_project-main\src\dataset"
    extractor = EmbeddingExtractor(dataset_path, "OpenFace")
    embeddings, labels = extractor.extract_embeddings()
    print(labels)
