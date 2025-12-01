# ASL Recognition System

A complete **American Sign Language (ASL)** hand gesture recognition system with real-time prediction using **MediaPipe** and **CNN (Convolutional Neural Network)**.

## 🎯 Features

- **36 Classes**: Recognizes 0-9 (digits) and a-z (letters)
- **MediaPipe Integration**: Hand detection and landmark tracking
- **Background Removal**: Automatic hand segmentation for improved accuracy
- **Real-Time Prediction**: Live webcam recognition with confidence scores
- **Data Augmentation**: Multiple augmentation techniques to prevent overfitting
- **CNN Architecture**: Deep learning model with batch normalization and dropout
- **Custom Data Collection**: Built-in tool to collect your own training data
- **Comprehensive Training Pipeline**: Includes callbacks, TensorBoard logging, and visualization

## 📁 Project Structure

```
asl_recognition-/
├── asl_dataset/              # Main dataset (36 classes)
├── custom_dataset/           # Your custom collected data
├── processed_data/           # Preprocessed numpy arrays
├── checkpoints/              # Best model checkpoints
├── logs/                     # TensorBoard training logs
├── models/                   # Saved trained models
├── data_preprocessing.py     # Dataset preprocessing with MediaPipe
├── data_collector.py         # Real-time data collection tool
├── cnn_model.py             # CNN model architecture
├── train.py                 # Basic training script
├── train_with_augmentation.py # Training with data augmentation
├── live_prediction.py       # Real-time ASL recognition
└── requirements.txt         # Python dependencies
```

## 🚀 Installation

1. **Clone the repository**
```bash
cd asl_recognition-
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preprocessing

The preprocessing pipeline uses **MediaPipe** for hand detection and background removal, ensuring consistency between training and real-time prediction.

### Run Preprocessing

```bash
python data_preprocessing.py
```

This will:
1. Load images from `asl_dataset/`
2. Apply MediaPipe hand detection
3. Remove background (set to black)
4. Convert to grayscale
5. Resize to 64x64 pixels
6. Normalize pixel values [0, 1]
7. Split into train/validation/test sets
8. Save processed arrays to `processed_data/`

### Output Files

- `processed_data/X_train.npy` - Training images
- `processed_data/X_val.npy` - Validation images
- `processed_data/X_test.npy` - Test images
- `processed_data/y_train.npy` - Training labels
- `processed_data/y_val.npy` - Validation labels
- `processed_data/y_test.npy` - Test labels
- `processed_data/metadata.json` - Dataset metadata
- `train_distribution.png` - Class distribution visualization
- `sample_images.png` - Sample preprocessed images

## 🎓 Training the Model

### Option 1: Basic Training

```bash
python train.py --architecture standard --epochs 50 --batch_size 32
```

### Option 2: Training with Data Augmentation (Recommended)

```bash
python train_with_augmentation.py --architecture standard --epochs 100 --batch_size 32 --learning_rate 0.0005
```

**Data augmentation includes:**
- Random rotation (±20°)
- Width/Height shift (±15%)
- Random zoom (±15%)
- Shear transformation (10%)

### Architecture Options

- `standard` - Balanced model (32-64-128 filters)
- `deep` - Deeper model with more layers (32-64-128-256 filters)
- `lightweight` - Faster, smaller model (16-32-64 filters)

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--architecture` | `standard` | Model architecture type |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--learning_rate` | `0.0005` | Learning rate for optimizer |

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## 🎥 Real-Time Prediction

Run the live prediction system with your webcam:

```bash
python live_prediction.py
```

**Features:**
- Real-time hand detection using MediaPipe
- Automatic background removal
- Prediction smoothing for stability
- Top-3 predictions with confidence scores
- Color-coded confidence levels:
  - 🟢 Green: High confidence (>70%)
  - 🟠 Orange: Medium confidence (50-70%)
  - 🔴 Red: Low confidence (<50%)

**Controls:**
- Press `Q` to quit

## 📷 Collect Custom Data

To collect your own training data:

```bash
python data_collector.py
```

**Usage:**
1. Enter the letter/digit you want to record (e.g., 'A')
2. Position your hand in the camera frame
3. Hold `SPACE` to capture images
4. Press `Q` to quit

**The tool automatically:**
- Detects your hand using MediaPipe
- Removes background
- Crops hand region
- Converts to grayscale
- Saves to `custom_dataset/`

## 🏗️ CNN Model Architecture

### Standard Model

```
Input (64x64x1 grayscale) 
    ↓
[Conv Block 1] 32 filters → BatchNorm → Conv 32 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
[Conv Block 2] 64 filters → BatchNorm → Conv 64 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
[Conv Block 3] 128 filters → BatchNorm → Conv 128 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Output(36) [Softmax]
```

**Key Features:**
- Batch Normalization for training stability
- Dropout layers to prevent overfitting
- Multiple convolutional blocks for feature extraction
- Categorical crossentropy loss
- Adam optimizer

## 📈 Results & Evaluation

After training, the following visualizations are generated:

1. **Training History** (`training_history_augmented.png`)
   - Accuracy curves (train vs validation)
   - Loss curves (train vs validation)

2. **Confusion Matrix** (`confusion_matrix_augmented.png`)
   - Shows prediction accuracy per class
   - Identifies commonly confused signs

3. **Model Checkpoints** (`checkpoints/`)
   - Best model saved based on validation accuracy

## 🛠️ Technical Details

### Preprocessing Pipeline

1. **MediaPipe Hand Detection**
   - Detects 21 hand landmarks
   - Creates bounding box around hand
   - Generates convex hull mask

2. **Background Removal**
   - Applies mask to isolate hand
   - Sets background to black
   - Dilates mask to include edges

3. **Image Processing**
   - BGR → RGB → Grayscale conversion
   - Resize to 64x64 pixels
   - Normalize to [0, 1]
   - Add channel dimension (64, 64, 1)

### Training Callbacks

- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if no improvement (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate when plateaued (patience=5)
- **TensorBoard**: Logs training metrics for visualization

## 🔧 Troubleshooting

### Model Loading Error

If you encounter `ValueError: Unrecognized keyword arguments: ['batch_shape']`, the code includes automatic compatibility handling for older Keras models.

### No Hand Detected

- Ensure good lighting conditions
- Keep hand within camera frame
- Adjust `min_detection_confidence` in MediaPipe settings

### Low Accuracy

- Collect more training data for problematic classes
- Increase data augmentation
- Train for more epochs
- Try different model architectures

## 📚 Usage Examples

### Load Preprocessed Data

```python
from data_preprocessing import ASLDataPreprocessor

preprocessor = ASLDataPreprocessor(dataset_path='asl_dataset')
X_train, X_val, X_test, y_train, y_val, y_test, metadata = \
    preprocessor.load_processed_data('processed_data')
```

### Train Custom Model

```python
from cnn_model import ASLCNNModel

# Initialize
model = ASLCNNModel(metadata_path='processed_data/metadata.json')

# Build and compile
model.build_model(architecture='standard')
model.compile_model(learning_rate=0.001, optimizer='adam')

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = model.load_data('processed_data')

# Train
history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

# Evaluate
test_metrics = model.evaluate(X_test, y_test)

# Save
model.save_model('models/my_model.keras')
```

### Make Predictions

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('checkpoints/best_model.keras')

# Prepare image (preprocessed to 64x64x1)
# ... preprocessing code ...

# Predict
predictions = model.predict(image_batch)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
```

## 🎯 Model Performance

Expected performance with proper training:

- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 90-95%
- **Test Accuracy**: 88-93%
- **Real-time FPS**: 15-30 fps (depending on hardware)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional ASL signs (phrases, words)
- Multi-hand detection
- Mobile deployment
- Web interface
- Real-time translation

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **TensorFlow/Keras** for deep learning framework
- ASL dataset contributors

## 📞 Support

For issues or questions:
1. Check existing issues
2. Review troubleshooting section
3. Create new issue with detailed description

---

**Happy Learning! 🤟 (ASL "I Love You" sign)**
