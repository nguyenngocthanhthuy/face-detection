import cv2
from face_lib import face_lib

# from ..Image_Comparer import


class FaceDetectionApp:
    def __init__(
        self,
    ):
        self.video_capture = cv2.VideoCapture(0)
        self.FL = face_lib()
        if not self.video_capture.isOpened():
            raise Exception("Error: Could not open video capture")

    def process_frame(self, frame):
        faces_detected = True
        frame_copy = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        while faces_detected:
            faces, boxes = self.FL.faces_locations(frame)
            if faces == 0:
                faces_detected = False
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
                cv2.putText(
                    frame_copy,
                    "person",
                    (int(x), int(y - 10)),  # Vị trí chữ
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font chữ
                    0.5,  # Kích thước chữ
                    (255, 255, 255),  # Màu chữ (trắng)
                    1,  # Độ dày chữ
                    cv2.LINE_AA,  # Kiểu chữ
                )

                # Xóa khuôn mặt đã phát hiện để tiếp tục phát hiện khuôn mặt khác
                # cv2.imwrite(f"{uuid.uuid4()}.jpg", frame[y : h, x : w])
                frame[y:h, x:w] = 0

        return frame_copy

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = self.process_frame(frame)
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceDetectionApp()
    app.run()
