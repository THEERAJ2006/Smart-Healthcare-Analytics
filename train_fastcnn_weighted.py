#!/usr/bin/env python3
"""FastCNN training with class weights to handle imbalance"""

import os
import json
import warnings
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20  # Increased from 10 for better convergence
MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n" + "="*65)
print("  FASTCNN TRAINING WITH CLASS WEIGHTS")
print("="*65)

# Build datasets
print("\n[Loading datasets...]")
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode="nearest"
)

val_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

train_ds = train_gen.flow_from_directory(
    "data/chest_xray/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_ds = val_test_gen.flow_from_directory(
    "data/chest_xray/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_ds = val_test_gen.flow_from_directory(
    "data/chest_xray/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_indices = train_ds.class_indices
class_names = {v: k for k, v in class_indices.items()}

print(f"[OK] Datasets loaded")
print(f"    Train: {train_ds.samples} images")
print(f"    Val:   {val_ds.samples} images")
print(f"    Test:  {test_ds.samples} images")
print(f"    Classes: {class_names}")

# Compute class weights
print("\n[Computing class weights...]")
# Get all labels from training set
y_train = []
train_ds.reset()
for _ in range(len(train_ds)):
    _, labels = next(train_ds)
    # Handle both tensor and numpy array formats
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
    y_train.extend(np.argmax(labels_np, axis=1))
    
y_train = np.array(y_train[:train_ds.samples])  # Ensure we have exactly train_ds.samples
train_ds.reset()  # Reset for actual training

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print("[OK] Class weights computed:")
for cls_idx, weight in class_weight_dict.items():
    print(f"    {class_names[cls_idx]}: {weight:.3f}")

# Build model
print("\n[Building FastCNN model...]")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(IMG_SIZE + (3,)),
    
    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),
    
    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),
    
    # Block 3 (added for more capacity)
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),
    
    # Head
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

print("[OK] Model compiled")
print(f"    Total parameters: {model.count_params():,}")

# Callbacks
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "FastCNN_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0
    )
]

# Train
print("\n[STAR] Starting training with class weights...")
print(f"    Epochs: {EPOCHS}")
print(f"    Batch size: {BATCH_SIZE}")
print("="*65)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
    class_weight=class_weight_dict,  # KEY: Apply class weights
    verbose=1
)

print("\n[OK] Training completed!")

# Plot history
print("\n[Saving training plots...]")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="Train", linewidth=2, color="#2ecc71")
axes[0].plot(history.history["val_accuracy"], label="Val", linewidth=2, color="#e74c3c")
axes[0].set_title("Accuracy (Weighted Training)", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Train", linewidth=2, color="#2ecc71")
axes[1].plot(history.history["val_loss"], label="Val", linewidth=2, color="#f39c12")
axes[1].set_title("Loss (Weighted Training)", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "FastCNN_training_history_weighted.png")
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"[OK] Plot saved -> {plot_path}")

print("\n[COMPLETE] FastCNN training finished with class weights!")
print(f"[OK] Model saved -> {os.path.join(MODEL_DIR, 'FastCNN_best.keras')}")
print("\nNext: Evaluate the new model with eval_fastcnn.py\n")
