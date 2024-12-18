import faiss


class FaissIndex:
    def __init__(self, embeddings, labels):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels = labels

    def save_index(self, index_path, labels_path):
        faiss.write_index(self.index, index_path)
        with open(labels_path, "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")

        print("Save weight successfully!")


if __name__ == "__main__":
    # Usage
    faiss_index = FaissIndex(1, 1)
    faiss_index.save_index("faiss_index.bin", "labels.txt")
