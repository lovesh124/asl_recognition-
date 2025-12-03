
import os
import numpy as np
import json
import cv2
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

class ASLAnnotator:
    
    def __init__(self, dataset_path):
        
        self.dataset_path = Path(dataset_path)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        print("ASL Annotator initialized")
        print(f"MediaPipe Hands loaded")
        print(f"{len(self.class_names)} classes to annotate")
    
    def calculate_image_quality(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # blur detection
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(lap_var / 500.0, 1.0)
        
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        
        # overall quality
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
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        
        landmarks = []
        x_coords = []
        yCoords = []
        
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
            yCoords.append(y)
        
        # bounding box with some padding
        xmin = max(0, min(x_coords) - 20)
        ymin = max(0, min(yCoords) - 20)
        xmax = min(w, max(x_coords) + 20)
        ymax = min(h, max(yCoords) + 20)  # +20 for padding
        
        bbox = {
            'x': int(xmin),
            'y': int(ymin),
            'width': int(xmax - xmin),
            'height': int(ymax - ymin)
        }
        
        hand_side = handedness.classification[0].label.lower()
        hand_confidence = float(handedness.classification[0].score)
        
        return {
            'hand_detected': True,
            'hand_side': hand_side,
            'hand_confidence': hand_confidence,
            'landmarks': landmarks,
            'bbox': bbox,
            'num_landmarks': len(landmarks)
        }
    
    def annotate_image(self, image_path, class_label):
        try:
            image = cv2.imread(str(image_path))
            
            if image is None:
                return {'filename': str(image_path), 'class': class_label, 'hand_detected': False}
            
            h, w = image.shape[:2]
            
            handAnnotations = self.extract_hand_annotations(image)
            quality = self.calculate_image_quality(image)
            
            annotation = {
                'filename': str(image_path.relative_to(self.dataset_path)),
                'absolute_path': str(image_path),
                'class': class_label,
                'image_width': w,
                'image_height': h,
                'quality_metrics': quality
            }
            
            if handAnnotations:
                annotation.update(handAnnotations)
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
        print("ANNOTATING ASL DATASET")
        
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
                print(f"Warning: Class folder '{class_name}' not found")
                continue
            
            image_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            if not image_files:
                print(f"No images found for class '{class_name}'")
                continue
            
            print(f"\nAnnotating class '{class_name}': {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"Class {class_name}"):
                annot = self.annotate_image(img_path, class_name)
                all_annotations['annotations'].append(annot)
                
                total_images += 1
                
                if 'error' in annot:
                    failed_annotations += 1
                elif not annot.get('hand_detected', False):
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
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        print(f"Total images processed: {total_images}")
        print(f"Successful annotations: {successful_annotations}")
        print(f"No hand detected: {no_hand_detected}")
        print(f"Failed annotations: {failed_annotations}")
        print(f"\nAnnotations saved to: {output_path.absolute()}")
        
        return all_annotations
    
    def load_annotations(self, annotation_file='asl_annotations.json'):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations['annotations'])} annotations")
        return annotations
    
    def get_class_annotations(self, annotations, class_name):
        return [ann for ann in annotations['annotations'] 
                if ann['class'] == class_name]
    
    def get_annotations_with_hand(self, annotations):
        return [ann for ann in annotations['annotations'] 
                if ann.get('hand_detected', False)]
    
    def cleanup(self):
        self.hands.close()


def main():
    DATASET_PATH = 'asl_dataset'
    OUTPUT_FILE = 'asl_annotations.json'
    
    print("ASL DATASET ANNOTATION TOOL")
    print("This will create rich annotations for your existing dataset:")
    print("  Hand landmarks (21 points with x, y, z coordinates)")
    print("  Bounding boxes (x, y, width, height)")
    print("  Hand side detection (left/right)")
    print("  Image quality metrics (blur, brightness, contrast)")
    
    annotator = ASLAnnotator(dataset_path=DATASET_PATH)
    
    annotations = annotator.annotate_dataset(output_file=OUTPUT_FILE)
    
    print("SAMPLE ANNOTATIONS")
    
    with_hand = annotator.get_annotations_with_hand(annotations)
    
    if with_hand:
        sample = with_hand[0]
        print(f"\nClass: {sample['class']}")
        print(f"File: {sample['filename']}")
        print(f"Hand Side: {sample.get('hand_side', 'N/A')}")
        print(f"Number of Landmarks: {sample.get('num_landmarks', 0)}")
        print(f"Bounding Box: {sample.get('bbox', 'N/A')}")
        print(f"Image Quality: {sample['quality_metrics']['overall_quality']:.3f}")
        print(f"   Blur Score: {sample['quality_metrics']['blur_score']:.3f}")
        print(f"   Brightness: {sample['quality_metrics']['brightness']:.3f}")
        print(f"   Contrast: {sample['quality_metrics']['contrast']:.3f}")
    
    annotator.cleanup()
    
    print("ANNOTATION COMPLETE!")
    print(f"\nAnnotations saved to: {OUTPUT_FILE}")
    print("\nYou can now use these annotations for:")
    print("  Enhanced model training with metadata")
    print("  Error analysis and debugging")
    print("  Dataset quality assessment")
    print("  Multi-task learning (predict class + hand side)")
    print("  Filtering low-quality images")



if __name__ == "__main__":
    main()
