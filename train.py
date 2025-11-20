"""
Simple Training Script for ASL CNN Model
Run this script to train the model with default or custom parameters
"""

from cnn_model import ASLCNNModel
import argparse


def train_asl_model(architecture='standard', epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the ASL recognition model
    
    Args:
        architecture: Model architecture ('standard', 'deep', 'lightweight')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    
    print("="*70)
    print("ASL HAND GESTURE RECOGNITION - TRAINING")
    print("="*70)
    print(f"Configuration:")
    print(f"  Architecture: {architecture}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print("="*70 + "\n")
    
    # Initialize model
    model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    
    # Build and compile model
    model.build_model(architecture=architecture)
    model.get_model_summary()
    model.compile_model(learning_rate=learning_rate, optimizer='adam')
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = model.load_data('processed_data')
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_callbacks=True
    )
    
    # Plot training history
    model.plot_training_history(save_path='training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
    
    # Save the trained model
    model.save_model('models/asl_cnn_final.keras')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print("\nGenerated files:")
    print("  ✓ training_history.png - Training accuracy and loss curves")
    print("  ✓ confusion_matrix.png - Confusion matrix heatmap")
    print("  ✓ models/asl_cnn_final.keras - Trained model")
    print("  ✓ checkpoints/ - Best model checkpoints during training")
    print("  ✓ logs/ - TensorBoard logs")
    print("\nTo view TensorBoard logs, run:")
    print("  tensorboard --logdir=logs")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASL CNN Model')
    
    parser.add_argument('--architecture', type=str, default='standard',
                        choices=['standard', 'deep', 'lightweight'],
                        help='Model architecture (default: standard)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Train the model
    train_asl_model(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
