

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


# CNN model for ASL recognition
class ASLCNNModel:
    def __init__(self, metadata_path='processed_data/metadata.json'):
        # load info about the dataset
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.input_shape = tuple(self.metadata['input_shape'])
        self.num_classes = self.metadata['num_classes']
        self.classNames = self.metadata['class_names']
        
        print(f"Input shape: {self.input_shape}")
        print(f"Num classes: {self.num_classes}")
        
        self.model = None
        self.history = None
    
    def build_model(self, architecture='standard'):
        if architecture == 'standard':  # this one is the best
            self.model = self._build_standard_model()
        elif architecture == 'deep':
            self.model = self._build_deep_model()
        elif architecture == 'lightweight':
            self.model = self._build_lightweight_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return self.model
    
    def _build_standard_model(self):
        # tried a bunch of different architectures, this one gave best results
        model = models.Sequential(name='ASL_CNN_Standard')
        
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        # first conv block
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'))
        model.add(layers.BatchNormalization(name='bn1_1'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.BatchNormalization(name='bn1_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.25, name='dropout1'))  # helps prevent overfitting
        
        # second block with more filters
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.BatchNormalization(name='bn2_1'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.BatchNormalization(name='bn2_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25, name='dropout2'))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.BatchNormalization(name='bn3_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.BatchNormalization(name='bn3_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.25, name='dropout3'))
        
        model.add(layers.Flatten(name='flatten'))
        
        model.add(layers.Dense(256, activation='relu', name='dense1'))
        model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        model.add(layers.Dense(128, activation='relu', name='dense2'))
        model.add(layers.BatchNormalization(name='bn_dense2'))
        model.add(layers.Dropout(0.5, name='dropout_dense2'))
        
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        return model
    
    def _build_deep_model(self):
        model = models.Sequential(name='ASL_CNN_Deep')
        
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'))
        model.add(layers.BatchNormalization(name='bn1_1'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.BatchNormalization(name='bn1_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.2, name='dropout1'))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.BatchNormalization(name='bn2_1'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.BatchNormalization(name='bn2_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25, name='dropout2'))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.BatchNormalization(name='bn3_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.BatchNormalization(name='bn3_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.3, name='dropout3'))
        
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'))
        model.add(layers.BatchNormalization(name='bn4_1'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'))
        model.add(layers.BatchNormalization(name='bn4_2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool4'))
        model.add(layers.Dropout(0.3, name='dropout4'))
        
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(512, activation='relu', name='dense1'))
        model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        model.add(layers.Dense(256, activation='relu', name='dense2'))
        model.add(layers.BatchNormalization(name='bn_dense2'))
        model.add(layers.Dropout(0.5, name='dropout_dense2'))
        
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        return model
    
    def _build_lightweight_model(self):
        model = models.Sequential(name='ASL_CNN_Lightweight')
        
        model.add(layers.Input(shape=self.input_shape, name='input_layer'))
        
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.2, name='dropout1'))
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.2, name='dropout2'))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.3, name='dropout3'))
        
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, activation='relu', name='dense1'))
        model.add(layers.Dropout(0.5, name='dropout_dense1'))
        
        model.add(layers.Dense(self.num_classes, activation='softmax', name='output_layer'))
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # compile model
        
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"\nTotal params: {total_params:,}")
        print(f"Input: {self.input_shape}, Output: {self.num_classes} classes")
    
    def load_data(self, data_dir='processed_data'):
        data_path = Path(data_dir)
        print(f"\nLoading data from {data_path}...")
        
        X_train = np.load(data_path / 'X_train.npy')
        X_val = np.load(data_path / 'X_val.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_train = np.load(data_path / 'y_train.npy')
        y_val = np.load(data_path / 'y_val.npy')
        y_test = np.load(data_path / 'y_test.npy')
        
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        print(f"Training: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, 
              use_callbacks=True,
              checkpoint_dir='checkpoints',
              log_dir='logs'):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() and compile_model() first.")
        
        print(f"\nTraining {len(X_train)} samples")
        
        callbacks = []
        if use_callbacks:
            callbacks = self._setup_callbacks(checkpoint_dir, log_dir)
        
        # train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def _setup_callbacks(self, checkpoint_dir='checkpoints', log_dir='logs'):
        Path(checkpoint_dir).mkdir(exist_ok=True)
        Path(log_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp for unique filenames
        callbacks = []  # list of callbacks to use during training
        
        checkpoint_path = f"{checkpoint_dir}/best_model_{timestamp}.keras"
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpoint: {checkpoint_path}")
        
        # stop if validation loss doesn't improve
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,  # wait 10 epochs before stopping
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        tensorboard_callback = TensorBoard(
            log_dir=f"{log_dir}/{timestamp}",
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard_callback)
        print(f"TensorBoard logs: {log_dir}/{timestamp}")
        print(f"Run: tensorboard --logdir={log_dir}")
        
        return callbacks
    
    def evaluate(self, X_test, y_test, verbose=1):
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        print("\nEvaluating on test set")
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            print(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict(self, X, return_probabilities=False):
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def plot_training_history(self, save_path='training_history.png'):
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.history.history
        epochs_range = range(1, len(history['loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # accuracy
        ax1.plot(epochs_range, history['accuracy'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs_range, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # loss
        ax2.plot(epochs_range, history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='confusion_matrix.png'):
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
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='models/asl_cnn_model.keras'):
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.load_weights(filepath)
        print(f"Loaded weights from {filepath}")


def main():
    print("ASL Hand Recognition Training")
    print("asl hand reconization")
    
    asl_model = ASLCNNModel(metadata_path='processed_data/metadata.json')
    asl_model.build_model(architecture='standard')
    asl_model.get_model_summary()
    asl_model.compile_model(learning_rate=0.001, optimizer='adam')
    
    X_train, X_val, X_test, y_train, y_val, y_test = asl_model.load_data('processed_data')
    
    history = asl_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        use_callbacks=True
    )
    
    # plot results
    asl_model.plot_training_history(save_path='training_history.png')
    test_metrics = asl_model.evaluate(X_test, y_test)
    asl_model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
    
    # save model
    asl_model.save_model('models/asl_cnn_final.keras')
    
    print("\nDone! Files generated:")
    print(" - training_history.png")
    print(" - confusion_matrix.png")
    print(" - models/asl_cnn_final.keras")
    print(" - checkpoints/ and logs/")


if __name__ == "__main__":
    main()
