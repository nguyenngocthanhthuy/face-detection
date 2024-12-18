import cv2
import os
from datetime import datetime
from face_lib import face_lib
from infer_img import ImageProcessor
import numpy as np

class FaceDetectionApp:
    def __init__(self, model_name, index_path, labels_path):
        self.FL = face_lib()
        self.processor = ImageProcessor(model_name, index_path, labels_path)
        self.video_capture = cv2.VideoCapture(0)
        self.output_dir = self.create_output_dir()
        self.video_writer = self.init_video_writer()
       

    def create_output_dir(self):
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("output_videos", start_time)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(self.output_dir, 'video.avi')
        frame_width = int(self.video_capture.get(3))
        frame_height = int(self.video_capture.get(4))
        return cv2.VideoWriter(out_path, fourcc, 20.0, (frame_width, frame_height))


    
    
    def process_frame(self, frame):

        frame_copy = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        for _ in range(2):
            faces, boxes = self.FL.faces_locations(frame)
            if faces > 0:
                for x, y, w, h in boxes:
                    # Chuyển tọa độ nhỏ hơn 0 thành 0 và giới hạn trong kích thước ảnh
                    x = max(0, x - 10)
                    y = max(0, y - 50)
                    w = min(frame_width, x + w + 15)
                    h = min(frame_height, y + h + 75)

                    cv2.rectangle(
                        frame_copy,
                        (int(x), int(y)),
                        (int(w), int(h)),
                        (22, 22, 250),
                        2,
                    )

                    # Ghi chữ "abc" lên box
                    img = frame[y:h, x:w]

                    label = self.processor.process_image(img)

                    cv2.putText(
                        frame_copy,
                        label,
                        (int(x), int(y - 10)),  # Vị trí chữ
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font chữ
                        0.5,  # Kích thước chữ
                        (255, 255, 255),  # Màu chữ (trắng)
                        1,  # Độ dày chữ
                        cv2.LINE_AA,  # Kiểu chữ
                    )
                    # Xóa khuôn mặt đã phát hiện để tiếp tục phát hiện khuôn mặt khác
                    frame[y:h, x:w] = 0
        return frame_copy
    

    

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = self.process_frame(frame)
            self.video_writer.write(frame)
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        self.video_capture.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_name = r"OpenFace"
    index_path = r"D:\doan2\sp_project-main\sp_project-main\src\face_embeddings\weights\faiss_index.bin"
    labels_path = r"D:\doan2\sp_project-main\sp_project-main\src\face_embeddings\weights\labels.txt"
    app = FaceDetectionApp(model_name, index_path, labels_path)
    app.run()
