from Image_Comparer.imagecomparer import ImageComparer
from face_embeddings.Extract_Embeddings.embeddingextractor import Embeddings, CropFace
import cv2


class ImageProcessor:
    def __init__(self, model_name, index_path, labels_path, n_top=3):
        self.model_name = model_name
        self.index_path = index_path
        self.labels_path = labels_path
        self.n_top = n_top
        self.img_comparer = ImageComparer(self.index_path, self.labels_path)
        self.crop_face = CropFace()
        self.embedding_extractor = Embeddings(self.model_name)

    def process_image(self, img_path):
        img_crop = self.crop_face.crop_face_img(img_path)
        embedding_vector = self.embedding_extractor.get_embeddings(img_crop)
        label = self.img_comparer.compare_image(embedding_vector, self.n_top)
        return label


if __name__ == "__main__":
    img_path = r"src\dataset\Dong\363430026_3401355540177335_2475338294135339786_n.jpg"
    processor = ImageProcessor(
        model_name=r"OpenFace",
        index_path=r"/home/nntthuy/face-detection/src/face_embeddings/weights/faiss_index.bin",
        labels_path=r"/home/nntthuy/face-detection/src/face_embeddings/weights/labels.txt",
    )
    label = processor.process_image(img_path)
    print(label)
