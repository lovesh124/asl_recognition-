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
from datetime import datetime
import os


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
    
    def load_data(self, data_dir='processed_data'):
        """
        Load preprocessed data from disk
        
        Args:
            data_dir: Directory containing preprocessed numpy files
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        data_path = Path(data_dir)
        
        print(f"\nLoading preprocessed data from {data_path}...")
        
        X_train = np.load(data_path / 'X_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_train = np.load(data_path / 'y_train.npy')
        y_val = np.load(data_path / 'y_val.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        # Convert labels to categorical (one-hot encoding)
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        print(f"✓ Data loaded successfully!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, 
              use_callbacks=True,
              checkpoint_dir='checkpoints',
              log_dir='logs'):
        """
        Train the CNN model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_callbacks: Whether to use training callbacks
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() and compile_model() first.")
        
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING")
        print("="*70)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {len(X_train) // batch_size}")
        print("="*70 + "\n")
        
        # Setup callbacks
        callbacks = []
        if use_callbacks:
            callbacks = self._setup_callbacks(checkpoint_dir, log_dir)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        
        return self.history
    
    def _setup_callbacks(self, checkpoint_dir='checkpoints', log_dir='logs'):
        """
        Setup training callbacks
        
        Args:
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
            
        Returns:
            List of callbacks
        """
        # Create directories
        Path(checkpoint_dir).mkdir(exist_ok=True)
        Path(log_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = []
        
        # ModelCheckpoint: Save best model
        checkpoint_path = f"{checkpoint_dir}/best_model_{timestamp}.keras"
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        print(f"✓ ModelCheckpoint: Will save best model to {checkpoint_path}")
        
        # EarlyStopping: Stop if no improvement
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        print(f"✓ EarlyStopping: Patience=10 epochs")
        
        # ReduceLROnPlateau: Reduce learning rate when plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"✓ ReduceLROnPlateau: Will reduce LR by 0.5 if no improvement for 5 epochs")
        
        # TensorBoard: Visualization
        tensorboard_callback = TensorBoard(
            log_dir=f"{log_dir}/{timestamp}",
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard_callback)
        print(f"✓ TensorBoard: Logs saved to {log_dir}/{timestamp}")
        print(f"  Run: tensorboard --logdir={log_dir}")
        
        return callbacks
    
    def evaluate(self, X_test, y_test, verbose=1):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity mode
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        
        # Get metric names and values
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            print(f"{name}: {value:.4f}")
        
        print("="*70)
        
        return metrics
    
    def predict(self, X, return_probabilities=False):
        """
        Make predictions on new data
        
        Args:
            X: Input images
            return_probabilities: If True, return probabilities; else return class indices
            
        Returns:
            Predictions (class indices or probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history (accuracy and loss curves)
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.history.history
        epochs_range = range(1, len(history['loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(epochs_range, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs_range, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(epochs_range, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            save_path: Path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='models/asl_cnn_model.keras'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model_weights(self, filepath):
        """
        Load model weights from file
        
        Args:
            filepath: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.load_weights(filepath)
        print(f"✓ Model weights loaded from {filepath}")


def main():
    """
    Main function to demonstrate complete training pipeline
    """
    print("="*70)
    print("ASL HAND GESTURE RECOGNITION - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Initialize model
    asl_model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    
    # Build model
    asl_model.build_model(architecture='standard')
    
    # Display model summary
    asl_model.get_model_summary()
    
    # Compile model
    asl_model.compile_model(learning_rate=0.001, optimizer='adam')
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = asl_model.load_data('processed_data')
    
    # Train model
    history = asl_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        use_callbacks=True
    )
    
    # Plot training history
    asl_model.plot_training_history(save_path='training_history.png')
    
    # Evaluate on test set
    test_metrics = asl_model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    asl_model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
    
    # Save the trained model
    asl_model.save_model('models/asl_cnn_final.keras')
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - training_history.png: Training curves")
    print("  - confusion_matrix.png: Confusion matrix")
    print("  - models/asl_cnn_final.keras: Trained model")
    print("  - checkpoints/: Best model checkpoints")
    print("  - logs/: TensorBoard logs")


if __name__ == "__main__":
    main()
