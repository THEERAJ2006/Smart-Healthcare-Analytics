#!/usr/bin/env python3
"""Fast evaluation script for FastCNN model"""

import os
import json
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
MODEL_PATH = "models/FastCNN_best.keras"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n" + "="*65)
print("  FASTCNN EVALUATION")
print("="*65)

# Load datasets
print("\n[Loading test dataset...]")
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
test_ds = test_gen.flow_from_directory(
    "data/chest_xray/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
print(f"[OK] Test set loaded: {test_ds.samples} images")

# Load model
print(f"\n[Loading model from {MODEL_PATH}...]")
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded")

# Evaluate
print("\n[Evaluating on test set...]")
test_ds.reset()

y_true = []
y_pred_prob = []

# Calculate steps needed to iterate through the entire dataset once
steps_per_epoch = np.ceil(test_ds.samples / BATCH_SIZE).astype(int)

for i, (images, labels) in enumerate(test_ds):
    if i >= steps_per_epoch:
        break
    if i % 10 == 0:
        print(f"  Processing batch {i+1}/{steps_per_epoch}...")
    
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
    y_true.extend(np.argmax(labels_np, axis=1))
    probs = model.predict(images, verbose=0)
    y_pred_prob.extend(probs)

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = np.argmax(y_pred_prob, axis=1)

# Compute metrics
accuracy = float(accuracy_score(y_true, y_pred))

# Ensure y_pred_prob sums to 1.0 for AUC calculation
y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
try:
    auc = float(roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average='weighted'))
except Exception as e:
    print(f"[!] Could not compute AUC: {e}")
    auc = 0.0

# Sensitivity per class
class_names = list(test_ds.class_indices.keys())
sensitivities = {}
for i, cls in enumerate(class_names):
    mask = y_true == i
    if np.any(mask):
        sensitivities[cls] = float(np.mean(y_pred[mask] == y_true[mask]))
    else:
        sensitivities[cls] = 0.0

avg_sensitivity = float(np.mean(list(sensitivities.values())))

# Print results
print("\n" + "="*65)
print("  FASTCNN EVALUATION RESULTS")
print("="*65)
print(f"  Accuracy    : {accuracy*100:.2f}%")
print(f"  AUC         : {auc:.4f}")
print(f"  Sensitivity : {avg_sensitivity*100:.2f}%")
print("  Per-class sensitivity:")
for cls, sens in sensitivities.items():
    print(f"    - {cls}: {sens*100:.2f}%")
print("="*65)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=class_names, yticklabels=class_names,
            cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('FastCNN - Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(PLOT_DIR, "FastCNN_confusion_matrix.png")
plt.savefig(cm_path, dpi=100)
plt.close()
print(f"\n[OK] Confusion matrix saved -> {cm_path}")

# Classification report
print("\n[Classification Report]")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save metrics
metrics = {
    "accuracy": accuracy,
    "auc": auc, 
    "sensitivity": avg_sensitivity,
    "sensitivities_per_class": sensitivities,
    "model_name": "FastCNN"
}

metrics_path = os.path.join("models", "FastCNN_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n[OK] Metrics saved -> {metrics_path}")

print("\n[COMPLETE] FastCNN evaluation finished!\n")
