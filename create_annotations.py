"""
ASL Dataset Annotation Script
Creates rich annotations for existing dataset using MediaPipe
Stores: class label, hand landmarks, bounding box, hand side, image quality
"""

import os
import numpy as np
import json
import cv2
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

class ASLAnnotator:
    """
    Creates annotations for ASL hand gesture images
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the annotator
        
        Args:
            dataset_path: Path to the asl_dataset folder
        """
        self.dataset_path = Path(dataset_path)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Define class names (0-9, a-z)
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        print("✓ ASL Annotator initialized")
        print(f"✓ MediaPipe Hands loaded")
        print(f"✓ {len(self.class_names)} classes to annotate")
    
    def calculate_image_quality(self, image):
        """
        Calculate image quality metrics
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for quality calculations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize to [0, 1]
        
        # 2. Brightness
        brightness = np.mean(gray) / 255.0
        
        # 3. Contrast (standard deviation)
        contrast = np.std(gray) / 128.0  # Normalize
        
        # 4. Overall quality score (weighted average)
        quality_score = (blur_score * 0.5 + 
                        (1 - abs(brightness - 0.5) * 2) * 0.3 + 
                        contrast * 0.2)
        
        return {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'overall_quality': float(quality_score)
        }
    
    def extract_hand_annotations(self, image):
        """
        Extract hand-related annotations using MediaPipe
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with hand annotations or None if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_handedness = results.multi_handedness[0]
        
        # Extract landmarks
        landmarks = []
        x_coords = []
        y_coords = []
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            
            landmarks.append({
                'id': idx,
                'x': float(x),
                'y': float(y),
                'z': float(z)
            })
            
            x_coords.append(x)
            y_coords.append(y)
        
        # Calculate bounding box
        x_min = max(0, min(x_coords) - 20)
        y_min = max(0, min(y_coords) - 20)
        x_max = min(w, max(x_coords) + 20)
        y_max = min(h, max(y_coords) + 20)
        
        bbox = {
            'x': int(x_min),
            'y': int(y_min),
            'width': int(x_max - x_min),
            'height': int(y_max - y_min)
        }
        
        # Get hand side (left or right)
        hand_side = hand_handedness.classification[0].label.lower()
        hand_confidence = float(hand_handedness.classification[0].score)
        
        return {
            'hand_detected': True,
            'hand_side': hand_side,
            'hand_confidence': hand_confidence,
            'landmarks': landmarks,
            'bbox': bbox,
            'num_landmarks': len(landmarks)
        }
    
    def annotate_image(self, image_path, class_label):
        """
        Create complete annotation for a single image
        
        Args:
            image_path: Path to the image file
            class_label: Class name/label for the image
            
        Returns:
            Dictionary with all annotations
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            
            if image is None:
                return {
                    'filename': str(image_path),
                    'class': class_label,
                    'error': 'Failed to load image',
                    'hand_detected': False
                }
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Extract hand annotations
            hand_annotations = self.extract_hand_annotations(image)
            
            # Calculate image quality
            quality_metrics = self.calculate_image_quality(image)
            
            # Build complete annotation
            annotation = {
                'filename': str(image_path.relative_to(self.dataset_path)),
                'absolute_path': str(image_path),
                'class': class_label,
                'image_width': w,
                'image_height': h,
                'quality_metrics': quality_metrics
            }
            
            # Add hand annotations if hand was detected
            if hand_annotations:
                annotation.update(hand_annotations)
            else:
                annotation['hand_detected'] = False
                annotation['landmarks'] = []
                annotation['bbox'] = None
            
            return annotation
            
        except Exception as e:
            return {
                'filename': str(image_path),
                'class': class_label,
                'error': str(e),
                'hand_detected': False
            }
    
    def annotate_dataset(self, output_file='asl_annotations.json'):
        """
        Annotate entire dataset
        
        Args:
            output_file: Path to save annotations JSON file
            
        Returns:
            Dictionary with all annotations
        """
        print("\n" + "="*70)
        print("ANNOTATING ASL DATASET")
        print("="*70)
        
        all_annotations = {
            'dataset_info': {
                'dataset_path': str(self.dataset_path),
                'num_classes': len(self.class_names),
                'classes': self.class_names
            },
            'annotations': []
        }
        
        total_images = 0
        successful_annotations = 0
        failed_annotations = 0
        no_hand_detected = 0
        
        # Process each class
        for class_name in self.class_names:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                print(f"⚠ Warning: Class folder '{class_name}' not found!")
                continue
            
            # Get all image files
            image_files = list(class_path.glob('*.jpeg')) + \
                         list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.png'))
            
            if not image_files:
                print(f"⚠ No images found for class '{class_name}'")
                continue
            
            print(f"\nAnnotating class '{class_name}': {len(image_files)} images")
            
            # Annotate each image with progress bar
            for img_path in tqdm(image_files, desc=f"Class {class_name}"):
                annotation = self.annotate_image(img_path, class_name)
                all_annotations['annotations'].append(annotation)
                
                total_images += 1
                
                if 'error' in annotation:
                    failed_annotations += 1
                elif not annotation.get('hand_detected', False):
                    no_hand_detected += 1
                else:
                    successful_annotations += 1
        
        # Add statistics
        all_annotations['statistics'] = {
            'total_images': total_images,
            'successful_annotations': successful_annotations,
            'no_hand_detected': no_hand_detected,
            'failed_annotations': failed_annotations
        }
        
        # Save annotations to JSON file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        print("\n" + "="*70)
        print("ANNOTATION SUMMARY")
        print("="*70)
        print(f"Total images processed: {total_images}")
        print(f"✓ Successful annotations: {successful_annotations}")
        print(f"⚠ No hand detected: {no_hand_detected}")
        print(f"✗ Failed annotations: {failed_annotations}")
        print(f"\n✓ Annotations saved to: {output_path.absolute()}")
        print("="*70)
        
        return all_annotations
    
    def load_annotations(self, annotation_file='asl_annotations.json'):
        """
        Load previously created annotations
        
        Args:
            annotation_file: Path to annotations JSON file
            
        Returns:
            Dictionary with all annotations
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"✓ Loaded {len(annotations['annotations'])} annotations")
        return annotations
    
    def get_class_annotations(self, annotations, class_name):
        """
        Get all annotations for a specific class
        
        Args:
            annotations: Full annotations dictionary
            class_name: Class to filter
            
        Returns:
            List of annotations for the class
        """
        return [ann for ann in annotations['annotations'] 
                if ann['class'] == class_name]
    
    def get_annotations_with_hand(self, annotations):
        """
        Get only annotations where hand was detected
        
        Args:
            annotations: Full annotations dictionary
            
        Returns:
            List of annotations with hand detected
        """
        return [ann for ann in annotations['annotations'] 
                if ann.get('hand_detected', False)]
    
    def cleanup(self):
        """Close MediaPipe resources"""
        self.hands.close()


def main():
    """
    Main function to annotate the ASL dataset
    """
    # Set dataset path
    DATASET_PATH = 'asl_dataset'
    OUTPUT_FILE = 'asl_annotations.json'
    
    print("="*70)
    print("ASL DATASET ANNOTATION TOOL")
    print("="*70)
    print("This will create rich annotations for your existing dataset:")
    print("  ✓ Hand landmarks (21 points with x, y, z coordinates)")
    print("  ✓ Bounding boxes (x, y, width, height)")
    print("  ✓ Hand side detection (left/right)")
    print("  ✓ Image quality metrics (blur, brightness, contrast)")
    print("="*70 + "\n")
    
    # Initialize annotator
    annotator = ASLAnnotator(dataset_path=DATASET_PATH)
    
    # Annotate entire dataset
    annotations = annotator.annotate_dataset(output_file=OUTPUT_FILE)
    
    # Print some sample annotations
    print("\n" + "="*70)
    print("SAMPLE ANNOTATIONS")
    print("="*70)
    
    # Get annotations with hand detected
    with_hand = annotator.get_annotations_with_hand(annotations)
    
    if with_hand:
        sample = with_hand[0]
        print(f"\nClass: {sample['class']}")
        print(f"File: {sample['filename']}")
        print(f"Hand Side: {sample.get('hand_side', 'N/A')}")
        print(f"Number of Landmarks: {sample.get('num_landmarks', 0)}")
        print(f"Bounding Box: {sample.get('bbox', 'N/A')}")
        print(f"Image Quality: {sample['quality_metrics']['overall_quality']:.3f}")
        print(f"  - Blur Score: {sample['quality_metrics']['blur_score']:.3f}")
        print(f"  - Brightness: {sample['quality_metrics']['brightness']:.3f}")
        print(f"  - Contrast: {sample['quality_metrics']['contrast']:.3f}")
    
    # Cleanup
    annotator.cleanup()
    
    print("\n" + "="*70)
    print("ANNOTATION COMPLETE!")
    print("="*70)
    print(f"\n✓ Annotations saved to: {OUTPUT_FILE}")
    print("\nYou can now use these annotations for:")
    print("  • Enhanced model training with metadata")
    print("  • Error analysis and debugging")
    print("  • Dataset quality assessment")
    print("  • Multi-task learning (predict class + hand side)")
    print("  • Filtering low-quality images")
    print("="*70)


if __name__ == "__main__":
    main()
