"""
ASL Dataset Preprocessing Script
Prepares the ASL hand gesture dataset for CNN model training
"""

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

class ASLDataPreprocessor:
    """
    Preprocessor class for ASL hand gesture recognition dataset
    """
    
    def __init__(self, dataset_path, img_size=(64, 64), test_size=0.2, val_size=0.1, random_state=42):
        """
        Initialize the preprocessor
        
        Args:
            dataset_path: Path to the asl_dataset folder
            img_size: Target image size (height, width)
            test_size: Proportion of dataset for testing
            val_size: Proportion of training set for validation
            random_state: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Define class names (0-9, a-z)
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        print(f"Initialized preprocessor for {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
    
    def load_and_preprocess_images(self, normalize=True, grayscale=False):
        """
        Load all images from the dataset and preprocess them
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert images to grayscale
            
        Returns:
            X: Array of preprocessed images
            y: Array of labels
            image_paths: List of image file paths
        """
        images = []
        labels = []
        image_paths = []
        
        print("Loading images from dataset...")
        
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
                    continue
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(X)}")
        print(f"Image shape: {X.shape}")
        print(f"Data type: {X.dtype}")
        
        return X, y, image_paths
    
    def encode_labels(self, labels):
        """
        Encode string labels to integers
        
        Args:
            labels: Array of string labels
            
        Returns:
            Encoded integer labels
        """
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        """
        Decode integer labels back to strings
        
        Args:
            encoded_labels: Array of integer labels
            
        Returns:
            String labels
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def split_dataset(self, X, y, stratify=True):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Image data
            y: Labels
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\nSplitting dataset...")
        
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
        """
        Get class distribution statistics
        
        Args:
            labels: Array of labels (can be encoded or not)
            
        Returns:
            Dictionary with class counts
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def visualize_samples(self, X, y, num_samples=16, save_path=None):
        """
        Visualize random samples from the dataset
        
        Args:
            X: Image data
            y: Labels (encoded)
            num_samples: Number of samples to display
            save_path: Path to save the visualization
        """
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
        """
        Plot class distribution
        
        Args:
            y: Labels (encoded)
            title: Plot title
            save_path: Path to save the plot
        """
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
        """
        Save processed data to disk
        
        Args:
            X_train, X_val, X_test: Image data splits
            y_train, y_val, y_test: Label splits
            output_dir: Directory to save processed data
        """
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
        """
        Load previously processed data
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, metadata
        """
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
    """
    Main function to demonstrate data preprocessing pipeline
    """
    # Set parameters
    DATASET_PATH = 'asl_dataset'
    IMG_SIZE = (64, 64)  # Adjust based on your needs
    GRAYSCALE = False  # Set to True if you want grayscale images
    
    # Initialize preprocessor
    preprocessor = ASLDataPreprocessor(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Load and preprocess images
    X, y, image_paths = preprocessor.load_and_preprocess_images(
        normalize=True,
        grayscale=GRAYSCALE
    )
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y, stratify=True)
    
    # Visualize class distribution
    print("\nClass distribution in training set:")
    train_dist = preprocessor.get_class_distribution(y_train)
    for label, count in sorted(train_dist.items()):
        decoded_label = preprocessor.decode_labels([label])[0]
        print(f"  Class '{decoded_label}': {count} images")
    
    # Plot class distribution
    preprocessor.plot_class_distribution(y_train, title="Training Set Class Distribution", save_path="train_distribution.png")
    
    # Visualize sample images
    preprocessor.visualize_samples(X_train, y_train, num_samples=16, save_path="sample_images.png")
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "="*50)
    print("Data preprocessing completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. The processed data is saved in 'processed_data/' directory")
    print("2. You can now use this data to train your CNN model")
    print("3. Load the data using: preprocessor.load_processed_data('processed_data')")


if __name__ == "__main__":
    main()
