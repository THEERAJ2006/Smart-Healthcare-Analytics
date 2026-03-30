#!/usr/bin/env python3
"""
Improved FastCNN training with hyperparameter tuning
- Better regularization
- Class weights for imbalanced data
- Enhanced architecture
- Lower learning rate
- Better data augmentation
"""

import os, json, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 32  # Reduced from 64 for better gradient updates
EPOCHS = 30  # Increased from 10
MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n" + "="*70)
print("  FASTCNN RETRAINING - IMPROVED HYPERPARAMETERS")
print("="*70)

# Load datasets with better augmentation
print("\n[Loading datasets...]")
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=25,  # Increased from 15
    width_shift_range=0.15,  # Increased from 0.1
    height_shift_range=0.15,  # Increased from 0.1
    horizontal_flip=True,
    zoom_range=0.2,  # Increased from 0.15
    shear_range=0.1,  # New
    fill_mode="nearest"
)

val_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

train_ds = train_gen.flow_from_directory(
    "data/chest_xray/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_ds = val_test_gen.flow_from_directory(
    "data/chest_xray/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_ds = val_test_gen.flow_from_directory(
    "data/chest_xray/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

classes = train_ds.class_indices
print(f"[OK] Classes: {classes}")
print(f"    Train: {train_ds.samples} | Val: {val_ds.samples} | Test: {test_ds.samples}")

# Calculate class weights to handle imbalance
class_counts = np.array([len(os.listdir(f"data/chest_xray/train/{c}")) for c in sorted(classes.keys())])
class_weights = compute_class_weight('balanced', 
                                      classes=np.arange(len(classes)),
                                      y=np.repeat(np.arange(len(class_counts)), class_counts))
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print(f"[OK] Class weights (to handle imbalance):")
for cls_name, idx in classes.items():
    print(f"    {cls_name}: {class_weights_dict[idx]:.4f}")

# Improved FastCNN with better architecture
def build_improved_fastcnn():
    """Enhanced CNN with better regularization and architecture."""
    model = models.Sequential([
        layers.Input(IMG_SIZE + (3,)),
        
        # Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu", 
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),  # Increased from 0.25
        
        # Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),  # Increased from 0.25
        
        # Block 3 (new)
        layers.Conv2D(128, 3, padding="same", activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        # Global pooling + dense head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax")
    ])
    
    # Lower learning rate for better convergence
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    return model

print("\n[Building improved FastCNN model...]")
model = build_improved_fastcnn()
print(f"[OK] Model built with {model.count_params():,} parameters")
model.summary()

# Callbacks
cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,  # Increased from 2
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "FastCNN_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

# Train with class weights
print("\n[Training FastCNN with improved hyperparameters...]")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: 0.0005")
print(f"  Epochs: {EPOCHS}")
print(f"  Regularization: L2 (0.0001)")
print("="*70)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=cbs,
    class_weight=class_weights_dict,
    verbose=1
)

print("\n[Training complete!]")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val", linewidth=2)
axes[0].set_title("Accuracy", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Train", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val", linewidth=2)
axes[1].set_title("Loss", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOT_DIR, "FastCNN_training_history.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"[OK] Training history -> {path}")

# Evaluate on test set
print("\n[Evaluating on test set...]")
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import seaborn as sns

test_ds.reset()
y_true = []
y_pred_prob = []

steps = np.ceil(test_ds.samples / BATCH_SIZE).astype(int)
for i, (images, labels) in enumerate(test_ds):
    if i >= steps:
        break
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
    y_true.extend(np.argmax(labels_np, axis=1))
    probs = model.predict(images, verbose=0)
    y_pred_prob.extend(probs)

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = float(accuracy_score(y_true, y_pred))
try:
    auc = float(roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average='weighted'))
except:
    auc = 0.0

# Per-class metrics
class_names = list(classes.keys())
sensitivities = {}
for i, cls in enumerate(class_names):
    mask = y_true == i
    if np.any(mask):
        sensitivities[cls] = float(np.mean(y_pred[mask] == y_true[mask]))
    else:
        sensitivities[cls] = 0.0

avg_sensitivity = float(np.mean(list(sensitivities.values())))

# Print results
print("\n" + "="*70)
print("  FASTCNN EVALUATION RESULTS (IMPROVED)")
print("="*70)
print(f"  Accuracy    : {accuracy*100:.2f}%")
print(f"  AUC         : {auc:.4f}")
print(f"  Sensitivity : {avg_sensitivity*100:.2f}%")
print("\n  Per-class sensitivity:")
for cls, sens in sensitivities.items():
    print(f"    - {cls}: {sens*100:.2f}%")
print("="*70)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('FastCNN (Improved) - Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(PLOT_DIR, "FastCNN_confusion_matrix.png")
plt.savefig(cm_path, dpi=100)
plt.close()
print(f"\n[OK] Confusion matrix -> {cm_path}")

# Classification report
print("\n[Classification Report]")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save metrics
metrics = {
    "accuracy": accuracy,
    "auc": auc,
    "sensitivity": avg_sensitivity,
    "sensitivities_per_class": sensitivities,
    "model_name": "FastCNN",
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": 0.0005,
        "regularization": "L2 (0.0001)",
        "dropout": [0.3, 0.5, 0.4],
        "architecture": "3 Conv blocks with class weights"
    }
}

metrics_path = os.path.join(MODEL_DIR, "FastCNN_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n[OK] Metrics saved -> {metrics_path}")

print("\n[COMPLETE] FastCNN retraining finished!\n")
