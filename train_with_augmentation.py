"""
Improved Training Script with Data Augmentation to Prevent Overfitting
"""

from cnn_model import ASLCNNModel
from data_augmentation import ASLDataAugmentor
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_asl_model_with_augmentation(architecture='standard', epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the ASL recognition model WITH DATA AUGMENTATION
    
    Args:
        architecture: Model architecture ('standard', 'deep', 'lightweight')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    
    print("="*70)
    print("ASL HAND GESTURE RECOGNITION - TRAINING WITH AUGMENTATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Architecture: {architecture}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Data Augmentation: ENABLED")
    print("="*70 + "\n")
    
    # Initialize model
    model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    
    # Build and compile model
    model.build_model(architecture=architecture)
    model.get_model_summary()
    model.compile_model(learning_rate=learning_rate, optimizer='adam')
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = model.load_data('processed_data')
    
    print("\n" + "="*70)
    print("APPLYING DATA AUGMENTATION")
    print("="*70)
    
    # Create data augmentation generator for GRAYSCALE images
    train_datagen = ImageDataGenerator(
        rotation_range=20,           # Rotate images randomly by 20 degrees
        width_shift_range=0.15,      # Shift images horizontally by 15%
        height_shift_range=0.15,     # Shift images vertically by 15%
        zoom_range=0.15,             # Random zoom
        shear_range=0.1,             # Shear transformation
        fill_mode='nearest'          # Fill missing pixels
    )
    
    # Validation data should NOT be augmented (only normalized)
    val_datagen = ImageDataGenerator()
    
    print("✓ Augmentation configured:")
    print("  - Random rotation: ±20°")
    print("  - Width/Height shift: ±15%")
    print("  - Random zoom: ±15%")
    print("  - Shear transformation: 10%")
    print("="*70 + "\n")
    
    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    # Train model with augmented data
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING WITH AUGMENTATION")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(X_train) // batch_size}")
    print("="*70 + "\n")
    
    # Setup callbacks
    callbacks = model._setup_callbacks('checkpoints', 'logs')
    
    # Train with generators
    history = model.model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=val_generator,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    model.history = type('obj', (object,), {'history': history.history})()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    # Plot training history
    model.plot_training_history(save_path='training_history_augmented.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix_augmented.png')
    
    # Save the trained model
    model.save_model('models/asl_cnn_augmented.keras')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print("\nGenerated files:")
    print("  ✓ training_history_augmented.png - Training curves")
    print("  ✓ confusion_matrix_augmented.png - Confusion matrix")
    print("  ✓ models/asl_cnn_augmented.keras - Trained model")
    print("  ✓ checkpoints/ - Best model checkpoints")
    print("  ✓ logs/ - TensorBoard logs")
    print("\nTo view TensorBoard logs, run:")
    print("  tensorboard --logdir=logs")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASL CNN Model with Data Augmentation')
    
    parser.add_argument('--architecture', type=str, default='standard',
                        choices=['standard', 'deep', 'lightweight'],
                        help='Model architecture (default: standard)')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    
    args = parser.parse_args()
    
    # Train the model
    train_asl_model_with_augmentation(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
