
import os


os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
import json
from collections import Counter

MODEL_PATH = "checkpoints/best_model_20251126_150135.keras"
METADATA_PATH = "processed_data/metadata.json"
IMG_SIZE = (64, 64)
PADDING = 40
CONFIDENCE_THRESHOLD = 0.3

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
class_names = metadata['class_names']

print("Loading model")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (ValueError, TypeError) as e:
    print(f"Direct load failed: {e}")
    print("Trying compatability mode")
    
    import keras
    from keras.layers import InputLayer
    
    class CompatibleInputLayer(InputLayer):
        def __init__(self, batch_shape=None, shape=None, **kwargs):
            if batch_shape is not None and shape is None:
                shape = batch_shape[1:]
            super().__init__(shape=shape, **kwargs)
    
    with keras.utils.custom_object_scope({'InputLayer': CompatibleInputLayer}):
        model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded!")
print(f"Classes: {len(class_names)}\n")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def get_square_bbox(landmarks, frame_w, frame_h, padding=PADDING):
    # get bounding box around hand
    x = [int(p.x * frame_w) for p in landmarks.landmark]
    y = [int(p.y * frame_h) for p in landmarks.landmark]
    
    x_min, x_max = min(x) - padding, max(x) + padding
    y_min, y_max = min(y) - padding, max(y) + padding
    
    width = x_max - x_min
    height = y_max - y_min
    
    # make it square so the model works better
    if width > height:
        y_min -= (width - height) // 2
        y_max += (width - height) // 2
    else:
        x_min -= (height - width) // 2
        x_max += (height - width) // 2
        
    return max(0, x_min), max(0, y_min), min(frame_w, x_max), min(frame_h, y_max)

def remove_background(frame, hand_landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    h, w = frame.shape[:2]
    points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

def preprocess_for_model(roi):
    # has to match exactly what we did in training or it won't work
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    roi_resized = cv2.resize(roi_gray, IMG_SIZE)
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    roi_processed = np.expand_dims(roi_normalized, axis=-1)
    roi_batch = np.expand_dims(roi_processed, axis=0)
    
    print(f"Input: {roi_batch.shape}, range: [{roi_batch.min():.3f}, {roi_batch.max():.3f}]")
    
    return roi_batch

def predict_sign(roi):
    input_data = preprocess_for_model(roi)
    pred = model.predict(input_data, verbose=0)
    
    predicted_class_idx = np.argmax(pred[0])
    confidence = pred[0][predicted_class_idx]
    predicted_label = class_names[predicted_class_idx]
    
    print(f"Prediction: {predicted_label} ({confidence:.3f})")
    
    top_3_indices = np.argsort(pred[0])[-3:][::-1]
    top_3_predictions = [(class_names[i], pred[0][i]) for i in top_3_indices]
    
    return predicted_label, confidence, top_3_predictions

print("Starting webcam.")
print("Press Q to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# these are for smoothing out the predictions so they don't jump around
prediction_history = []
HISTORY_SIZE = 5  # avg over last 5 frames

while True:
    ret, frm = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frm, 1)
    h, w, _ = frame.shape
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    displayFrame = frame.copy()
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x1, y1, x2, y2 = get_square_bbox(hand_landmarks, w, h)
        cv2.rectangle(displayFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        frame_no_bg = remove_background(frame, hand_landmarks)
        roi = frame_no_bg[y1:y2, x1:x2]
        
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            predicted_label, confidence, top_3 = predict_sign(roi)
            
            prediction_history.append(predicted_label)
            if len(prediction_history) > HISTORY_SIZE:
                prediction_history.pop(0)
            
            most_common = Counter(prediction_history).most_common(1)[0][0]
            
            # color code by confidence
            if confidence > 0.7:
                color = (0, 255, 0)
            elif confidence > 0.5:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            text = f"{most_common.upper()} ({confidence*100:.1f}%)"
            cv2.putText(displayFrame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            y_offset = 30
            for i, (label, conf) in enumerate(top_3):
                text = f"{i+1}. {label.upper()}: {conf*100:.1f}%"
                cv2.putText(displayFrame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            roi_preview = cv2.resize(roi, (150, 150))
            displayFrame[10:160, w-160:w-10] = roi_preview
    else:
        cv2.putText(displayFrame, "No hand detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prediction_history.clear()
    
    cv2.putText(displayFrame, "Press Q to quit", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("ASL Recognition", displayFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nclosed")
