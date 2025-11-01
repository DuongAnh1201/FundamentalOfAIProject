from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
class FaceDetection:
    def __init__(self):
        pass



    def face_live(self):
        # 1. Download and load the YOLO face detection model
        model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
        model = YOLO(model_path)

        # 2. Open webcam (0 = default camera)
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


        if not cap.isOpened():
            print("❌ Error: Cannot access the webcam.")
            return
        # Optional: set camera resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("🎥 Starting live face detection. Press 'q' to quit.")

        # 3. Read frames continuously
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame.")
                break

            # Run YOLO inference (reduce size for faster, smoother results)
            results = model.predict(frame, imgsz=320, conf=0.5, verbose=False)

            # Plot results on the frame
            annotated_frame = results[0].plot()

            # Show the frame
            cv2.imshow("Live Face Detection", annotated_frame)

            # Small delay (controls refresh rate, prevents flicker)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    m = FaceDetection()
    m.face_live()






