from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import os

from NewtonMethod import CustomLogisticRegression


class FaceDetection:
    def __init__(self):
        """
        Initialize YOLO face detector, ViT feature extractor, and
        emotion classification model (iteration 2).
        """
        # Device for ViT and logistic regression
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load YOLO face detection model (once)
        print("Loading YOLO face detection model...")
        model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt"
        )
        self.face_model = YOLO(model_path)
        print("YOLO model loaded.")

        # 2. Load ViT feature extractor
        print("Loading ViT feature extractor...")
        vit_name = "google/vit-base-patch16-224-in21k"
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_name)
        self.vit_model = ViTModel.from_pretrained(vit_name).to(self.device)
        self.vit_model.eval()
        print(f"ViT model loaded on {self.device}.")

        # 3. Load emotion classification model (iteration 2)
        print("Loading emotion classification model (iteration 2)...")
        self.emotion_model = CustomLogisticRegression(input_size=768, num_classes=7)

        # Path to the trained model from iteration 2
        iter2_model_path = os.path.join("models", "emotion_model_iter2.pth")
        if not os.path.exists(iter2_model_path):
            print(
                f"⚠️ Warning: Trained model for iteration 2 not found at {iter2_model_path}."
            )
            print("   Make sure you have trained and saved 'emotion_model_iter2.pth'.")
        else:
            self.emotion_model.load_weights(iter2_model_path)

        # Emotion label mapping (IDs 0-6)
        self.emotion_names = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Sad",
            "Surprised",
            "Neutral",
        ]

    def _extract_vit_features_from_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a 768-dim feature vector from a cropped face (BGR image).

        Args:
            face_bgr: Cropped face image from OpenCV (BGR format)

        Returns:
            features: numpy array of shape (1, 768)
        """
        # Convert BGR (OpenCV) to RGB and then to PIL image
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Preprocess with ViT processor
        inputs = self.vit_processor(images=face_pil, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.vit_model(**inputs)

        # CLS token: shape [1, 768]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.cpu().numpy()

    def _predict_emotion(self, features: np.ndarray) -> (int, str, float):
        """
        Predict emotion from a 768-dim feature vector.

        Args:
            features: numpy array of shape (1, 768)

        Returns:
            pred_id: predicted class id
            label: emotion label string
            confidence: max softmax probability
        """
        # Use the CustomLogisticRegression forward method
        # Ensure we always work with a NumPy array here
        with torch.no_grad():
            probs_tensor = self.emotion_model.forward(features)  # torch.Tensor

        # Convert to NumPy explicitly to avoid torch.max / axis issues
        probs = probs_tensor.detach().cpu().numpy()  # shape: (1, num_classes)

        pred_id = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs, axis=1)[0])

        # Safety check in case of label mismatch length
        if 0 <= pred_id < len(self.emotion_names):
            label = self.emotion_names[pred_id]
        else:
            label = f"Class {pred_id}"

        return pred_id, label, confidence

    def face_live(self):
        # 1. Open webcam (0 = default camera)
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


        if not cap.isOpened():
            print("❌ Error: Cannot access the webcam.")
            return
        # Optional: set camera resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("🎥 Starting live face detection with emotion labeling. Press 'q' to quit.")

        # 3. Read frames continuously
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame.")
                break

            # Run YOLO inference (reduce size for faster, smoother results)
            results = self.face_model.predict(frame, imgsz=320, conf=0.5, verbose=False)

            # Start from original frame and draw our own boxes + labels
            annotated_frame = frame.copy()

            if len(results) > 0:
                det = results[0]
                if det.boxes is not None and len(det.boxes) > 0:
                    boxes = det.boxes.xyxy.cpu().numpy()  # (N, 4)

                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int)

                        # Clamp to frame boundaries
                        h, w, _ = frame.shape
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Crop face
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        # Extract ViT features and predict emotion
                        try:
                            features = self._extract_vit_features_from_crop(face_crop)
                            # Predict emotion
                            _, label, conf = self._predict_emotion(features)
                            text = f"{label} ({conf*100:.1f}%)"
                        except Exception as e:
                            # In case of any error, log it and show generic label
                            print("Emotion prediction error:", e)
                            text = "Face"

                        # Draw bounding box
                        cv2.rectangle(
                            annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )

                        # ----- Draw label INSIDE the top of the bounding box -----
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )

                        # Background rectangle for text (inside the box)
                        label_x1 = x1
                        label_y1 = y1
                        label_x2 = x1 + text_width + 8
                        label_y2 = y1 + text_height + baseline + 6

                        # Clamp to frame bounds
                        h, w, _ = annotated_frame.shape
                        label_x1 = max(0, min(label_x1, w - 1))
                        label_x2 = max(0, min(label_x2, w - 1))
                        label_y1 = max(0, min(label_y1, h - 1))
                        label_y2 = max(0, min(label_y2, h - 1))

                        cv2.rectangle(
                            annotated_frame,
                            (label_x1, label_y1),
                            (label_x2, label_y2),
                            (0, 255, 0),
                            thickness=-1,  # filled
                        )

                        # Draw text on top of the filled rectangle
                        text_org = (
                            label_x1 + 3,
                            label_y1 + text_height + baseline // 2,
                        )
                        cv2.putText(
                            annotated_frame,
                            text,
                            text_org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),  # black text on green background
                            2,
                            cv2.LINE_AA,
                        )

            # Show the frame
            cv2.imshow("Live Face Detection + Emotion", annotated_frame)

            # Small delay (controls refresh rate, prevents flicker)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    m = FaceDetection()
    m.face_live()






