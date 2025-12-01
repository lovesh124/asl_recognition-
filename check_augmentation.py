"""
Check if model was trained with augmentation by analyzing training logs
"""
import os
import json
from pathlib import Path

print("="*70)
print("CHECKING IF MODEL WAS TRAINED WITH AUGMENTATION")
print("="*70)

# Check 1: Look for augmented model files
print("\n1. Checking for augmented model files...")
if Path("models/asl_cnn_augmented.keras").exists():
    print("   ✓ Found: models/asl_cnn_augmented.keras")
    augmented_model = True
else:
    print("   ✗ NOT FOUND: models/asl_cnn_augmented.keras")
    augmented_model = False

# Check 2: Look for augmented training visualizations
print("\n2. Checking for augmented training visualizations...")
if Path("training_history_augmented.png").exists():
    print("   ✓ Found: training_history_augmented.png")
else:
    print("   ✗ NOT FOUND: training_history_augmented.png")

if Path("confusion_matrix_augmented.png").exists():
    print("   ✓ Found: confusion_matrix_augmented.png")
else:
    print("   ✗ NOT FOUND: confusion_matrix_augmented.png")

# Check 3: Analyze TensorBoard logs
print("\n3. Analyzing TensorBoard logs...")
logs_dir = Path("logs")
if logs_dir.exists():
    runs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
    print(f"   Found {len(runs)} training runs:")
    for run in runs:
        print(f"   - {run.name}")
else:
    print("   ✗ No logs directory found")

# Check 4: Examine the checkpoint being used
print("\n4. Checking current checkpoint...")
checkpoint_path = Path("checkpoints/best_model_20251126_150135.keras")
if checkpoint_path.exists():
    print(f"   ✓ Using: {checkpoint_path.name}")
    print(f"   Created: Nov 26, 2024 at 15:01:35")
else:
    print("   ✗ Checkpoint not found")

# Final verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)
if augmented_model:
    print("✓ Model WAS trained with data augmentation")
    print("  Evidence: Augmented model file exists")
else:
    print("✗ Model was NOT trained with data augmentation")
    print("  Evidence:")
    print("  - No augmented model file (asl_cnn_augmented.keras)")
    print("  - No augmented training visualizations")
    print("  - Checkpoint is from basic training (train.py)")
    
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("To train with augmentation and improve model performance:")
print("  1. Run: python train_with_augmentation.py")
print("  2. Or run: python train_with_augmentation.py --epochs 100")
print("="*70)
