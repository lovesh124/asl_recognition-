"""
ASL Real-Time Prediction
Uses trained model with MediaPipe hand detection and background removal
"""

import os

# Force CPU path for MediaPipe in headless environments
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
import json
from collections import Counter

# CONFIGURATION
# Use latest checkpoint from training (Nov 26 run)
MODEL_PATH = "checkpoints/best_model_20251126_150135.keras"
METADATA_PATH = "processed_data/metadata.json"
IMG_SIZE = (64, 64)
PADDING = 40
CONFIDENCE_THRESHOLD = 0.3  # Lowered from 0.5 to see more predictions

# Load metadata
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
class_names = metadata['class_names']

# Load trained model
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded successfully!")
print(f"Classes: {class_names}\n")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def get_square_bbox(landmarks, frame_w, frame_h, padding=PADDING):
    """Get square bounding box around hand"""
    x = [int(p.x * frame_w) for p in landmarks.landmark]
    y = [int(p.y * frame_h) for p in landmarks.landmark]
    
    x_min, x_max = min(x) - padding, max(x) + padding
    y_min, y_max = min(y) - padding, max(y) + padding
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width > height:
        y_min -= (width - height) // 2
        y_max += (width - height) // 2
    else:
        x_min -= (height - width) // 2
        x_max += (height - width) // 2
        
    return max(0, x_min), max(0, y_min), min(frame_w, x_max), min(frame_h, y_max)

def remove_background(frame, hand_landmarks):
    """
    Remove background and keep only the hand region.
    Sets background to black (matching training data).
    """
    # Create mask from hand landmarks
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Get all landmark points
    h, w = frame.shape[:2]
    points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    
    # Create convex hull around hand
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    
    # Fill the hull with white (hand region)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Dilate mask to include hand edges
    kernel = np.ones((10, 10), np.uint8)  # Increased from (5,5) to (10,10)
    mask = cv2.dilate(mask, kernel, iterations=3)  # Increased from 2 to 3
    
    # Apply mask: keep hand, make background black
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

def preprocess_for_model(roi):
    """Preprocess the ROI to match training data format EXACTLY"""
    # Convert BGR to RGB first (matching training pipeline)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Then convert RGB to grayscale (matching training pipeline)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    roi_resized = cv2.resize(roi_gray, IMG_SIZE)
    
    # Normalize to [0, 1]
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    
    # Add channel dimension (64, 64, 1)
    roi_processed = np.expand_dims(roi_normalized, axis=-1)
    
    # Add batch dimension (1, 64, 64, 1)
    roi_batch = np.expand_dims(roi_processed, axis=0)
    
    # Debug: Print shape and value range
    print(f"Input shape: {roi_batch.shape}, Min: {roi_batch.min():.3f}, Max: {roi_batch.max():.3f}")
    
    return roi_batch

def predict_sign(roi):
    """Make prediction on the preprocessed ROI"""
    # Preprocess
    input_data = preprocess_for_model(roi)
    
    # Predict
    predictions = model.predict(input_data, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_label = class_names[predicted_class_idx]
    
    # Debug: Print prediction details
    print(f"Predicted: {predicted_label} with confidence: {confidence:.3f}")
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[i], predictions[0][i]) for i in top_3_indices]
    
    return predicted_label, confidence, top_3_predictions

# Start webcam
print("Starting webcam...")
print("Press 'Q' to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# Variables for smoothing predictions
prediction_history = []
HISTORY_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    display_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box
        x1, y1, x2, y2 = get_square_bbox(hand_landmarks, w, h)
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Remove background
        frame_no_bg = remove_background(frame, hand_landmarks)
        
        # Crop ROI
        roi = frame_no_bg[y1:y2, x1:x2]
        
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            # Make prediction
            predicted_label, confidence, top_3 = predict_sign(roi)
            
            # Add to history for smoothing
            prediction_history.append(predicted_label)
            if len(prediction_history) > HISTORY_SIZE:
                prediction_history.pop(0)
            
            # Get most common prediction in history
            most_common = Counter(prediction_history).most_common(1)[0][0]
            
            # Display prediction - ALWAYS show it, even with low confidence
            # Main prediction with color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            text = f"{most_common.upper()} ({confidence*100:.1f}%)"
            cv2.putText(display_frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Display top 3 predictions
            y_offset = 30
            for i, (label, conf) in enumerate(top_3):
                text = f"{i+1}. {label.upper()}: {conf*100:.1f}%"
                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Show preprocessed ROI (for debugging)
            roi_preview = cv2.resize(roi, (150, 150))
            display_frame[10:160, w-160:w-10] = roi_preview
    else:
        # No hand detected
        cv2.putText(display_frame, "No hand detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prediction_history.clear()
    
    # Instructions
    cv2.putText(display_frame, "Press 'Q' to quit", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("ASL Real-Time Recognition", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✓ Application closed")
