# training script for ASL model

# main training script
from cnn_model import ASLCNNModel
import argparse
from pathlib import Path


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    # looks for most recent checkpoint to resume training
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        ckpt_dir.glob("best_model_*.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def train_asl_model(architecture='standard', epochs=50, batch_size=32, learning_rate=0.001, resume_from=None):
    print(f"Training ASL model ({architecture})")
    print(f"epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    
    model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    model.build_model(architecture=architecture)
    model.get_model_summary()
    model.compile_model(learning_rate=learning_rate, optimizer='adam')

    # check if we should resume from a checkpoint
    checkpoint_to_load = Path(resume_from) if resume_from else find_latest_checkpoint()
    if checkpoint_to_load and checkpoint_to_load.exists():
        model.load_model_weights(str(checkpoint_to_load))
        print(f"Resuming from: {checkpoint_to_load}")
    else:
        print("Training from scratch")
    
    X_train, X_val, X_test, y_train, y_val, y_test = model.load_data('processed_data')
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_callbacks=True
    )
    
    model.plot_training_history(save_path='training_history.png')
    
    print("\nEvaluating...")
    test_results = model.evaluate(X_test, y_test)
    
    print("\nGenerating confusion matrix...")
    model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
    
    model.save_model('models/asl_cnn_final.keras')
    
    
    print(f"Accuracy: {test_results['accuracy']*100:.2f}%, Loss: {test_results['loss']:.4f}")
    print(f"Model saved to models/asl_cnn_final.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASL CNN Model')
    
    parser.add_argument('--architecture', type=str, default='standard',
                        choices=['standard', 'deep', 'lightweight'],
                        help='Model architecture')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train_asl_model(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from
    )
