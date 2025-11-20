"""
Data Augmentation Utilities for ASL Dataset
Provides augmentation techniques to increase dataset diversity
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from pathlib import Path


class ASLDataAugmentor:
    """
    Data augmentation class for ASL hand gesture images
    """
    
    def __init__(self):
        """Initialize augmentation pipelines"""
        pass
    
    def get_keras_augmentor(self, 
                           rotation_range=15,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           horizontal_flip=False,
                           brightness_range=(0.8, 1.2),
                           fill_mode='nearest'):
        """
        Get Keras ImageDataGenerator for augmentation
        
        Args:
            rotation_range: Degree range for random rotations
            width_shift_range: Fraction of total width for horizontal shifts
            height_shift_range: Fraction of total height for vertical shifts
            zoom_range: Range for random zoom
            horizontal_flip: Whether to randomly flip images horizontally
            brightness_range: Range for random brightness adjustment
            fill_mode: Points outside boundaries are filled according to mode
            
        Returns:
            ImageDataGenerator object
        """
        augmentor = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            brightness_range=brightness_range,
            fill_mode=fill_mode
        )
        
        return augmentor
    
    def get_albumentations_augmentor(self, img_size=(64, 64)):
        """
        Get Albumentations augmentation pipeline
        
        Args:
            img_size: Target image size
            
        Returns:
            Albumentations Compose object
        """
        augmentor = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            )
        ])
        
        return augmentor
    
    def augment_batch(self, images, augmentor, num_augmented=5):
        """
        Apply augmentation to a batch of images
        
        Args:
            images: Array of images
            augmentor: Albumentations augmentor
            num_augmented: Number of augmented versions per image
            
        Returns:
            Augmented images array
        """
        augmented_images = []
        
        for img in images:
            # Add original image
            augmented_images.append(img)
            
            # Generate augmented versions
            for _ in range(num_augmented):
                # Convert to uint8 if normalized
                if img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img.astype(np.uint8)
                
                # Apply augmentation
                augmented = augmentor(image=img_uint8)['image']
                
                # Convert back to float if original was normalized
                if img.max() <= 1.0:
                    augmented = augmented.astype(np.float32) / 255.0
                
                augmented_images.append(augmented)
        
        return np.array(augmented_images)
    
    def apply_simple_augmentations(self, image):
        """
        Apply simple augmentations without external libraries
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1 if image.max() <= 1 else 255)
        
        # Random translation
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            tx = int(np.random.uniform(-0.1, 0.1) * w)
            ty = int(np.random.uniform(-0.1, 0.1) * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random zoom
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h))
            
            if zoom_factor > 1:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = resized[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, 
                                          pad_w, w-new_w-pad_w, 
                                          cv2.BORDER_CONSTANT, value=0)
        
        return image


def create_augmented_dataset(X_train, y_train, augmentor, augmentations_per_image=2):
    """
    Create an augmented dataset from training data
    
    Args:
        X_train: Training images
        y_train: Training labels
        augmentor: ASLDataAugmentor instance
        augmentations_per_image: Number of augmented versions per image
        
    Returns:
        X_augmented, y_augmented: Augmented dataset
    """
    X_augmented = []
    y_augmented = []
    
    print(f"Creating augmented dataset with {augmentations_per_image} augmentations per image...")
    
    for i, (img, label) in enumerate(zip(X_train, y_train)):
        # Add original
        X_augmented.append(img)
        y_augmented.append(label)
        
        # Add augmented versions
        for _ in range(augmentations_per_image):
            aug_img = augmentor.apply_simple_augmentations(img.copy())
            X_augmented.append(aug_img)
            y_augmented.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(X_train)} images...")
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"\nAugmentation complete!")
    print(f"Original size: {len(X_train)}")
    print(f"Augmented size: {len(X_augmented)}")
    
    return X_augmented, y_augmented


if __name__ == "__main__":
    print("Data augmentation utilities loaded.")
    print("Import this module to use augmentation in your training pipeline.")
