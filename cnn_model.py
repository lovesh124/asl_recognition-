"""
CNN Model for ASL Hand Gesture Recognition
Implements a Convolutional Neural Network for classifying 36 ASL signs (0-9, a-z)
"""

import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


class ASLCNNModel:
    """
    CNN Model for ASL hand gesture recognition
    """
    
    def __init__(self, metadata_path='processed_data/metadata.json'):
        """
        Initialize the CNN model
        
        Args:
            metadata_path: Path to metadata.json file containing dataset information
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.input_shape = tuple(self.metadata['input_shape'])
        self.num_classes = self.metadata['num_classes']
        self.class_names = self.metadata['class_names']
        
        print(f"Initialized ASL CNN Model")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        
        self.model = None
        self.history = None
    
    def build_model(self, architecture='standard'):
        """
        Build the CNN model architecture
        
        Args:
            architecture: Model architecture type ('standard', 'deep', 'lightweight')
        """
        print(f"\nBuilding {architecture} CNN model...")
        
        if architecture == 'standard':
            self.model = self._build_standard_model()
        elif architecture == 'deep':
            self.model = self._build_deep_model()
        elif architecture == 'lightweight':
            self.model = self._build_lightweight_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        print("Model built successfully!")
        return self.model
    
    def _build_standard_model(self):
        """
        Build a standard CNN architecture
        
        Architecture:
        - Input Layer: Accepts images of shape (64, 64, 3)
        - Conv Block 1: 32 filters
        - Conv Block 2: 64 filters
        - Conv Block 3: 128 filters
        - Dense layers with dropout
        - Output: 36 classes with softmax
        """
        model = models.Sequential(name='ASL_CNN_Standard')
        
        # ============================================
        # INPUT LAYER
        # ============================================
        # Accepts RGB images of size 64x64x3
        # Input shape: (batch_size, 64, 64, 3)
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        print(f"✓ Input Layer added: shape={self.input_shape}")
        
        # ============================================
        # CONVOLUTIONAL BLOCK 1
        # ============================================
        # First convolutional layer: 32 filters, 3x3 kernel
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'))
        model.add(layers.BatchNormalization(name='bn1_1'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.BatchNormalization(name='bn1_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.25, name='dropout1'))
        
        print("✓ Conv Block 1 added: 32 filters")
        
        # ============================================
        # CONVOLUTIONAL BLOCK 2
        # ============================================
        # Second convolutional layer: 64 filters
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.BatchNormalization(name='bn2_1'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.BatchNormalization(name='bn2_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25, name='dropout2'))
        
        print("✓ Conv Block 2 added: 64 filters")
        
        # ============================================
        # CONVOLUTIONAL BLOCK 3
        # ============================================
        # Third convolutional layer: 128 filters
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.BatchNormalization(name='bn3_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.BatchNormalization(name='bn3_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.25, name='dropout3'))
        
        print("✓ Conv Block 3 added: 128 filters")
        
        # ============================================
        # FLATTEN LAYER
        # ============================================
        # Flatten the 3D output to 1D for dense layers
        model.add(layers.Flatten(name='flatten'))
        
        print("✓ Flatten layer added")
        
        # ============================================
        # FULLY CONNECTED LAYERS
        # ============================================
        # Dense layer 1: 256 neurons
        model.add(layers.Dense(256, activation='relu', name='dense1'))
        model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        # Dense layer 2: 128 neurons
        model.add(layers.Dense(128, activation='relu', name='dense2'))
        model.add(layers.BatchNormalization(name='bn_dense2'))
        model.add(layers.Dropout(0.5, name='dropout_dense2'))
        
        print("✓ Dense layers added: 256 and 128 neurons")
        
        # ============================================
        # OUTPUT LAYER
        # ============================================
        # Output layer: 36 neurons (one for each class) with softmax activation
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        print(f"✓ Output layer added: {self.num_classes} classes with softmax activation")
        
        return model
    
    def _build_deep_model(self):
        """
        Build a deeper CNN architecture with more layers
        """
        model = models.Sequential(name='ASL_CNN_Deep')
        
        # Input Layer
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        # Conv Block 1: 32 filters
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'))
        model.add(layers.BatchNormalization(name='bn1_1'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.BatchNormalization(name='bn1_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.2, name='dropout1'))
        
        # Conv Block 2: 64 filters
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.BatchNormalization(name='bn2_1'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.BatchNormalization(name='bn2_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25, name='dropout2'))
        
        # Conv Block 3: 128 filters
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.BatchNormalization(name='bn3_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.BatchNormalization(name='bn3_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.3, name='dropout3'))
        
        # Conv Block 4: 256 filters
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'))
        model.add(layers.BatchNormalization(name='bn4_1'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'))
        model.add(layers.BatchNormalization(name='bn4_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool4'))
        model.add(layers.Dropout(0.3, name='dropout4'))
        
        # Flatten and Dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(512, activation='relu', name='dense1'))
        model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        model.add(layers.Dense(256, activation='relu', name='dense2'))
        model.add(layers.BatchNormalization(name='bn_dense2'))
        model.add(layers.Dropout(0.5, name='dropout_dense2'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        return model
    
    def _build_lightweight_model(self):
        """
        Build a lightweight CNN architecture for faster training/inference
        """
        model = models.Sequential(name='ASL_CNN_Lightweight')
        
        # Input Layer
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        # Conv Block 1: 16 filters
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.2, name='dropout1'))
        
        # Conv Block 2: 32 filters
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.2, name='dropout2'))
        
        # Conv Block 3: 64 filters
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.3, name='dropout3'))
        
        # Flatten and Dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, activation='relu', name='dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            learning_rate: Learning rate for the optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        print(f"\nCompiling model with {optimizer} optimizer (lr={learning_rate})...")
        
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("✓ Model compiled successfully!")
        print(f"  - Loss: categorical_crossentropy")
        print(f"  - Metrics: accuracy, top_k_categorical_accuracy")
    
    def get_model_summary(self):
        """
        Display model architecture summary
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        self.model.summary()
        print("="*70)
        
        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Output Classes: {self.num_classes}")


def main():
    """
    Main function to demonstrate model creation
    """
    print("="*70)
    print("ASL HAND GESTURE RECOGNITION - CNN MODEL")
    print("="*70)
    
    # Initialize model
    asl_model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    
    # Build model (you can choose: 'standard', 'deep', or 'lightweight')
    asl_model.build_model(architecture='standard')
    
    # Display model summary
    asl_model.get_model_summary()
    
    # Compile model
    asl_model.compile_model(learning_rate=0.001, optimizer='adam')
    
    print("\n" + "="*70)
    print("Model is ready for training!")
    print("="*70)
    print("\nNext steps:")
    print("1. Load your preprocessed data")
    print("2. Train the model using model.train()")
    print("3. Evaluate on test set")


if __name__ == "__main__":
    main()
