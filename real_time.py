"""
FINAL Real-Time ASL Tracking
Compatible with: On-The-Fly Augmentation Model + Rescaling Layer
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque

class ASLHandTracker:
    def __init__(self, model_path='models/asl_cnn_final.keras', metadata_path='processed_data/metadata.json'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.prediction_buffer = deque(maxlen=5)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self._load_resources()

    def _load_resources(self):
        try:
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
                self.class_names = meta['class_names']
                
                # Handle input shape format
                if 'input_shape' in meta:
                    self.img_size = tuple(meta['input_shape'][:2])
                else:
                    self.img_size = (64, 64)
                    
            print("✓ Model loaded successfully.")
            print(f"✓ Input Size: {self.img_size}")
            print(f"✓ Classes: {len(self.class_names)}")
        except Exception as e:
            print(f"Error: {e}")
            exit(1)

    def apply_enhancements(self, image):
        """Must match the pipeline exactly: CLAHE Only"""
        # REMOVED: image = cv2.GaussianBlur(image, (5, 5), 0)

        # CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return final_img

    def get_square_bbox(self, landmarks, frame_w, frame_h, padding=40):
        """Calculates a square bounding box around the hand"""
        x_min, y_min = frame_w, frame_h
        x_max, y_max = 0, 0

        for lm in landmarks.landmark:
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_w, x_max + padding)
        y_max = min(frame_h, y_max + padding)

        width = x_max - x_min
        height = y_max - y_min
        
        # Force square aspect ratio
        if width > height:
            diff = width - height
            y_min = max(0, y_min - diff // 2)
            y_max = min(frame_h, y_max + diff // 2)
        else:
            diff = height - width
            x_min = max(0, x_min - diff // 2)
            x_max = min(frame_w, x_max + diff // 2)

        return x_min, y_min, x_max, y_max

    def run(self):
        # Force DirectShow for Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        print("\n=== READY ===")
        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw Skeleton
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )

                    try:
                        x1, y1, x2, y2 = self.get_square_bbox(hand_landmarks, w, h)
                        
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            
                            # 1. Enhance
                            roi_enhanced = self.apply_enhancements(roi)

                            # 2. Resize
                            img = cv2.resize(roi_enhanced, self.img_size)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # 3. Format for Model
                            # NO DIVISION BY 255 HERE! (Model handles it)
                            img = np.expand_dims(img, axis=0)

                            # 4. Predict
                            preds = self.model.predict(img, verbose=0)[0]
                            idx = np.argmax(preds)
                            confidence = preds[idx]
                            label = self.class_names[idx]
                            
                            # 5. Smooth Result
                            self.prediction_buffer.append((label, confidence))
                            labels = [p[0] for p in self.prediction_buffer]
                            final_label = max(set(labels), key=labels.count)

                            # 6. Draw UI
                            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Text Background
                            (text_w, text_h), _ = cv2.getTextSize(f"{final_label}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            cv2.rectangle(frame, (x1, y1-30), (x1 + text_w, y1), color, -1)
                            
                            cv2.putText(frame, f"{final_label}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            cv2.putText(frame, f"{confidence:.0%}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                    except Exception as e:
                        pass

            cv2.imshow("ASL Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ASLHandTracker()
    tracker.run()