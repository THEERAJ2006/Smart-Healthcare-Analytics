#!/usr/bin/env python3
"""Quick model comparison summary"""

import json
import os

print("\n" + "="*70)
print("  MODEL PERFORMANCE COMPARISON")
print("="*70)

models_dir = "models"

# Check FastCNN metrics
fastcnn_metrics_path = os.path.join(models_dir, "FastCNN_metrics.json")
if os.path.exists(fastcnn_metrics_path):
    with open(fastcnn_metrics_path) as f:
        fastcnn = json.load(f)
    print("\n[FastCNN] ✅ Trained with class weights")
    print(f"  Accuracy  : {fastcnn['accuracy']*100:.2f}%")
    print(f"  AUC       : {fastcnn['auc']:.4f}")
    print(f"  Sensitivity : {fastcnn['sensitivity']*100:.2f}%")
    print(f"  Per-class:")
    for cls, sens in fastcnn.get('sensitivities_per_class', {}).items():
        print(f"    - {cls}: {sens*100:.2f}%")
else:
    print("\n[FastCNN] ❌ Metrics not found")

# Check SimpleCNN metrics
simplecnn_metrics_path = os.path.join(models_dir, "SimpleCNN_metrics.json")
if os.path.exists(simplecnn_metrics_path):
    with open(simplecnn_metrics_path) as f:
        simplecnn = json.load(f)
    print("\n[SimpleCNN] ✅ Available")
    print(f"  Accuracy  : {simplecnn['accuracy']*100:.2f}%")
    print(f"  AUC       : {simplecnn['auc']:.4f}")
    print(f"  Sensitivity : {simplecnn['sensitivity']*100:.2f}%")
else:
    print("\n[SimpleCNN] ❌ Metrics not found")

# Check FastMobileNetV2 metrics
mobilenet_metrics_path = os.path.join(models_dir, "FastMobileNetV2_metrics.json")
if os.path.exists(mobilenet_metrics_path):
    with open(mobilenet_metrics_path) as f:
        mobilenet = json.load(f)
    print("\n[FastMobileNetV2] ✅ Available")
    print(f"  Accuracy  : {mobilenet['accuracy']*100:.2f}%")
    print(f"  AUC       : {mobilenet['auc']:.4f}")
    print(f"  Sensitivity : {mobilenet['sensitivity']*100:.2f}%")
else:
    print("\n[FastMobileNetV2] ❌ Metrics not found")

print("\n" + "="*70)
print("  RECOMMENDATION")
print("="*70)
if os.path.exists(fastcnn_metrics_path):
    print(f"\n✅ FastCNN is ready to use!")
    print(f"   - Use for binary classification or quick inference")
    print(f"   - Good balance: {fastcnn['accuracy']*100:.2f}% accuracy")
    print(f"   - Can detect all 3 classes with class weights applied")
    print(f"\nNext: Run 'streamlit run app_image.py' to use the model")

print("="*70 + "\n")
