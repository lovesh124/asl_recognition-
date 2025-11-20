# ASL Recognition Dataset Preprocessing

This project contains data preprocessing utilities for the ASL (American Sign Language) hand gesture recognition dataset.

## Dataset Structure
- 36 classes: 0-9 (digits) and a-z (letters)
- Images are organized in folders by class name
- Multiple images per class with different orientations and hand positions

## Features
- **Data Loading**: Automatic loading from folder structure
- **Preprocessing**: Resizing, normalization, grayscale conversion
- **Train/Val/Test Split**: Stratified splitting to maintain class distribution
- **Label Encoding**: String to integer label conversion
- **Visualization**: Class distribution plots and sample images
- **Data Augmentation**: Multiple augmentation techniques to improve model robustness
- **Save/Load**: Efficient numpy format for processed data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Preprocessing

```python
from data_preprocessing import ASLDataPreprocessor

# Initialize preprocessor
preprocessor = ASLDataPreprocessor(
    dataset_path='asl_dataset',
    img_size=(64, 64),  # Adjust based on your CNN architecture
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

# Load and preprocess images
X, y, image_paths = preprocessor.load_and_preprocess_images(
    normalize=True,
    grayscale=False
)

# Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X, y)

# Save processed data
preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
```

### Run Complete Pipeline

```bash
python data_preprocessing.py
```

This will:
1. Load all images from the dataset
2. Preprocess and resize them
3. Split into train/validation/test sets
4. Generate visualization plots
5. Save processed data to `processed_data/` directory

### Load Previously Processed Data

```python
from data_preprocessing import ASLDataPreprocessor

preprocessor = ASLDataPreprocessor(dataset_path='asl_dataset')
X_train, X_val, X_test, y_train, y_val, y_test, metadata = preprocessor.load_processed_data('processed_data')
```

### Data Augmentation

```python
from data_augmentation import ASLDataAugmentor, create_augmented_dataset

# Initialize augmentor
augmentor = ASLDataAugmentor()

# Create augmented dataset
X_train_aug, y_train_aug = create_augmented_dataset(
    X_train, y_train, 
    augmentor, 
    augmentations_per_image=2
)
```

## Output Files

After running the preprocessing:
- `processed_data/X_train.npy` - Training images
- `processed_data/X_val.npy` - Validation images
- `processed_data/X_test.npy` - Test images
- `processed_data/y_train.npy` - Training labels
- `processed_data/y_val.npy` - Validation labels
- `processed_data/y_test.npy` - Test labels
- `processed_data/metadata.json` - Dataset metadata
- `train_distribution.png` - Class distribution visualization
- `sample_images.png` - Sample images from dataset

## Configuration Options

### Image Size
Adjust `img_size` parameter based on your CNN architecture:
- Smaller (32x32, 64x64): Faster training, less memory
- Larger (128x128, 224x224): Better feature extraction, more memory

### Grayscale vs Color
- `grayscale=True`: Single channel, faster processing
- `grayscale=False`: RGB channels, preserves color information

### Data Splits
- `test_size`: Proportion for test set (default: 0.2)
- `val_size`: Proportion of training for validation (default: 0.1)

## Next Steps

After preprocessing, you can:
1. Train a CNN model using the processed data
2. Apply data augmentation for better generalization
3. Experiment with different image sizes and architectures
4. Use the metadata for model configuration

## Example CNN Input Shape

For a typical CNN model:
```python
input_shape = (64, 64, 3)  # For color images
# or
input_shape = (64, 64, 1)  # For grayscale images

num_classes = 36  # 0-9 and a-z
```
