"""
Fast Data Collector
Captures training data using MediaPipe cropping.
Saves directly to 'custom_dataset/' folder.
"""

import cv2
import os
import mediapipe as mp
import time
import numpy as np
from datetime import datetime

# CONFIGURATION
OUTPUT_DIR = "custom_dataset"
PADDING = 40

# Setup - Enable hand segmentation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def get_square_bbox(landmarks, frame_w, frame_h, padding=PADDING):
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
    
    # Dilate mask slightly to include hand edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply mask: keep hand, make background black
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

# Ask for label
label = input("Enter the letter you are recording (e.g. 'A'): ").upper()
save_path = os.path.join(OUTPUT_DIR, label)
os.makedirs(save_path, exist_ok=True)

# Check existing count
existing_count = len(os.listdir(save_path))
print(f"\n=== RECORDING CLASS: {label} ===")
print(f"Existing images: {existing_count}")
print("HOLD [SPACE] to capture.")
print("Press [Q] to quit.\n")

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    display_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x1, y1, x2, y2 = get_square_bbox(lm, w, h)
        
        # Draw Box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # CAPTURE LOGIC
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if x2 > x1 and y2 > y1:
                # Remove background from the frame
                frame_no_bg = remove_background(frame, lm)
                
                # Crop the region of interest
                roi = frame_no_bg[y1:y2, x1:x2]
                
                # Convert to grayscale (matching training data)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Save the processed image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{save_path}/{label}_{timestamp}.jpg"
                cv2.imwrite(filename, roi_gray)
                
                existing_count += 1
                cv2.putText(display_frame, "RECORDING", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # UI
    cv2.putText(display_frame, f"Count: {existing_count}", (10, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Data Collector", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
