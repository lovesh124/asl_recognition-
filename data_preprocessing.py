

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
        self.use_bg_removal = use_background_removal
        
        if self.use_bg_removal:
            # mediapipe for hand detection
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            print("MediaPipe initialized")
        
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.class_names)
        
        print(f"Preprocessor ready for {len(self.class_names)} classes")
        print(f"Background removal: {'ON' if use_background_removal else 'OFF'}")
    
    def remove_background(self, image):
        # this function removes the background so model focuses on hand
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return image
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = image.shape[:2]
        
        pts = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            pts.append([x, y])
        
        pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def load_and_preprocess_images(self, normalize=True, grayscale=False):
        imgs = []
        labels = []
        image_paths = []  
        skipped = 0
        
        print("Loading images")  
        
        for class_name in self.class_names:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class folder '{class_name}' not found")
                continue
            
            image_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            print(f"Class '{class_name}': {len(image_files)} images")
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        skipped += 1
                        continue
                    
                    if self.use_bg_removal:
                        img = self.remove_background(img)
                        if np.sum(img) == 0:
                            skipped += 1
                            continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    if grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    
                    img = cv2.resize(img, self.img_size)
                    
                    if normalize:  # always normalize for neural networks
                        img = img.astype(np.float32) / 255.0  # scale to 0-1
                    
                    imgs.append(img)
                    labels.append(class_name)
                    image_paths.append(str(img_path))
                    
                except Exception as e:
                    skipped += 1
                    continue
        
        img_data = np.array(imgs)
        y = np.array(labels)
        
        print(f"\nLoaded {len(img_data)} images (skipped {skipped})")
        print(f"Shape: {img_data.shape}, dtype: {img_data.dtype}")
        
        return img_data, y, image_paths

    def encode_labels(self, labels):
        return self.labelEncoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        return self.labelEncoder.inverse_transform(encoded_labels)
    
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
        decoded_labels = self.decode_labels(list(class_dist.keys()))
        counts = list(class_dist.values())
        
        plt.figure(figsize=(20, 6))
        plt.bar(decoded_labels, counts, color='skyblue', edgecolor='navy')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir='processed_data'):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving to {output_path}")
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
        
        print("Saved!")
        print(f"Location: {output_path.absolute()}")
    
    def load_processed_data(self, data_dir='processed_data'):
        data_path = Path(data_dir)
        print(f"Loading from {data_path}...")
        
        X_train = np.load(data_path / 'X_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_train = np.load(data_path / 'y_train.npy')
        y_val = np.load(data_path / 'y_val.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        with open(data_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("Data loaded successfully")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def main():
    DATASET_PATH = 'asl_dataset'
    IMG_SIZE = (64, 64)
    GRAYSCALE = True
    USE_BACKGROUND_REMOVAL = True
    
    preprocessor = ASLDataPreprocessor(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        use_background_removal=USE_BACKGROUND_REMOVAL
    )
    
    X, y, image_paths = preprocessor.load_and_preprocess_images(
        normalize=True,
        grayscale=GRAYSCALE
    )
    
    if GRAYSCALE and len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
        print(f"Added channel dim: {X.shape}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y, stratify=True)
    
    print("\nClass distribution in training set:")
    train_dist = preprocessor.get_class_distribution(y_train)
    for label, count in sorted(train_dist.items()):
        decoded_label = preprocessor.decode_labels([label])[0]
        print(f"  Class '{decoded_label}': {count} images")
    
    preprocessor.plot_class_distribution(y_train, title="Training Set Class Distribution (With Background Removal)", save_path="train_distribution.png")
    
    preprocessor.visualize_samples(X_train, y_train, num_samples=16, save_path="sample_images.png")
    
    preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    if preprocessor.use_bg_removal:
        preprocessor.hands.close()
    
    print("\nPreprocessing done!")
    print("Next: run train.py to train the model")


if __name__ == "__main__":
    main()
