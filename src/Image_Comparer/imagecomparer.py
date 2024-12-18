from deepface import DeepFace
import numpy as np
import faiss
from collections import Counter
import time


class ImageComparer:
    def __init__(self, index_path, labels_path):

        self.faiss_index = faiss.read_index(index_path)
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f]

    def compare_image(self, embedding, n_top):
        embedding = np.array([embedding]).astype("float32")
        _, indices = self.faiss_index.search(embedding, n_top)
        labels = [self.labels[idx] for idx in indices[0]]

        most_common_label = Counter(labels).most_common(1)[0][0]
        return most_common_label
