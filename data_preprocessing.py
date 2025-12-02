

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from pathlib import Path
import json
import mediapipe as mp

class ASLDataPreprocessor:
    
    def __init__(self, dataset_path, img_size=(64, 64), test_size=0.2, val_size=0.1, random_state=42, use_background_removal=True):
       
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_background_removal = use_background_removal
        
        # Initialize MediaPipe Hands if background removal is enabled
        if self.use_background_removal:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            print("✓ MediaPipe Hands initialized for background removal")
        
        # Define class names (0-9, a-z)
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        print(f"Initialized preprocessor for {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
        print(f"Background removal: {'ENABLED' if use_background_removal else 'DISABLED'}")
    
    def remove_background(self, image):
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            # No hand detected, return original image
            return image
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = image.shape[:2]
        
        # Get all landmark points
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
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        # Apply mask: keep hand, make background black
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def load_and_preprocess_images(self, normalize=True, grayscale=False):
        
        images = []
        labels = []
        image_paths = []
        
        skipped_count = 0
        
        print("Loading images from dataset")
        
        for class_name in self.class_names:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class folder '{class_name}' not found!")
                continue
            
            image_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            print(f"Loading {len(image_files)} images for class '{class_name}'...")
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        print(f"Warning: Could not load {img_path}")
                        skipped_count += 1
                        continue
                    
                    # Apply background removal if enabled
                    if self.use_background_removal:
                        img = self.remove_background(img)
                        
                        # Check if image is completely black (no hand detected)
                        if np.sum(img) == 0:
                            skipped_count += 1
                            continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Convert to grayscale if requested
                    if grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    
                    # Resize image
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalize if requested
                    if normalize:
                        img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    image_paths.append(str(img_path))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    skipped_count += 1
                    continue
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(X)}")
        print(f"Skipped images: {skipped_count}")
        print(f"Image shape: {X.shape}")
        print(f"Data type: {X.dtype}")
        
        return X, y, image_paths

    def encode_labels(self, labels):
       
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
       
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def split_dataset(self, X, y, stratify=True):
        
        print("\nSplitting dataset")
        
        # Encode labels
        y_encoded = self.encode_labels(y)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded if stratify else None
        )
        
        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp if stratify else None
        )
        
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, labels):
        
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def visualize_samples(self, X, y, num_samples=16, save_path=None):
        
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img_idx = indices[idx]
                img = X[img_idx]
                label = self.decode_labels([y[img_idx]])[0]
                
                # Handle grayscale images
                if len(img.shape) == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                
                ax.set_title(f"Class: {label}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, y, title="Class Distribution", save_path=None):
        
        class_dist = self.get_class_distribution(y)
        
        # Decode labels for plotting
        decoded_labels = self.decode_labels(list(class_dist.keys()))
        counts = list(class_dist.values())
        
        plt.figure(figsize=(20, 6))
        plt.bar(decoded_labels, counts, color='skyblue', edgecolor='navy')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir='processed_data'):
       
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving processed data to {output_path}...")
        
        # Save data as numpy arrays
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'X_val.npy', X_val)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'y_val.npy', y_val)
        np.save(output_path / 'y_test.npy', y_test)
        
        # Save metadata
        metadata = {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'img_size': self.img_size,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'input_shape': list(X_train.shape[1:])
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("Data saved successfully!")
        print(f"Files saved in: {output_path.absolute()}")
    
    def load_processed_data(self, data_dir='processed_data'):
        
        data_path = Path(data_dir)
        
        print(f"Loading processed data from {data_path}...")
        
        X_train = np.load(data_path / 'X_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_train = np.load(data_path / 'y_train.npy')
        y_val = np.load(data_path / 'y_val.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        with open(data_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("Data loaded successfully!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def main():
    
    # Set parameters
    DATASET_PATH = 'asl_dataset'
    IMG_SIZE = (64, 64)
    GRAYSCALE = True
    USE_BACKGROUND_REMOVAL = True  # Enable background removal to match live prediction
    
    # Initialize preprocessor
    preprocessor = ASLDataPreprocessor(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        use_background_removal=USE_BACKGROUND_REMOVAL
    )
    
    # Load and preprocess images
    X, y, image_paths = preprocessor.load_and_preprocess_images(
        normalize=True,
        grayscale=GRAYSCALE
    )
    
    # Add channel dimension for grayscale images if needed
    if GRAYSCALE and len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
        print(f"Added channel dimension. New shape: {X.shape}")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y, stratify=True)
    
    # Visualize class distribution
    print("\nClass distribution in training set:")
    train_dist = preprocessor.get_class_distribution(y_train)
    for label, count in sorted(train_dist.items()):
        decoded_label = preprocessor.decode_labels([label])[0]
        print(f"  Class '{decoded_label}': {count} images")
    
    # Plot class distribution
    preprocessor.plot_class_distribution(y_train, title="Training Set Class Distribution (With Background Removal)", save_path="train_distribution.png")
    
    # Visualize sample images
    preprocessor.visualize_samples(X_train, y_train, num_samples=16, save_path="sample_images.png")
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Close MediaPipe
    if preprocessor.use_background_removal:
        preprocessor.hands.close()
    
    
    print("Data preprocessing with background removal completed!")
  
    print("\nNext steps:")
    print("1. The processed data is saved in 'processed_data/' directory")
    print("2. Run train.py to retrain your model with background-removed images")
    print("3. The new model will match your live prediction preprocessing")


if __name__ == "__main__":
    main()
