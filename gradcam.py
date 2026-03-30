# =============================================================
#  GRAD-CAM — Visual Explanation for X-Ray Predictions
#  File: gradcam.py
#  Highlights WHICH regions of the X-ray influenced the decision
# =============================================================

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_last_conv_layer(model: tf.keras.Model) -> str:
    """Auto-detect the last Conv2D layer name in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # For MobileNetV2 — last conv is inside the base model
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given image array.

    Args:
        img_array        : preprocessed image (1, H, W, 3), float32 0–1
        model            : trained Keras model
        last_conv_layer_name : name of the last conv layer (auto-detect if None)

    Returns:
        heatmap : (H, W) float32 array, values 0–1
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    # Build a model that outputs [last_conv_output, predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification, use the single output neuron
        loss = predictions[:, 0]

    # Gradients of loss w.r.t. last conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_img : (H, W, 3) uint8 RGB image
        heatmap      : (h, w) float32 Grad-CAM heatmap
        alpha        : overlay transparency (0=invisible, 1=full)
        colormap     : OpenCV colormap

    Returns:
        overlaid : (H, W, 3) uint8 RGB image with heatmap overlay
    """
    h, w = original_img.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 0–255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    overlaid = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlaid


def generate_gradcam_figure(
    original_img: np.ndarray,
    model: tf.keras.Model,
    img_size: tuple = (224, 224),
    prediction_label: str = "",
    confidence: float = 0.0,
) -> plt.Figure:
    """
    Full pipeline: preprocess → Grad-CAM → overlay → matplotlib figure.
    Returns a matplotlib Figure ready for Streamlit display.
    """
    # Prepare input
    resized = cv2.resize(original_img, img_size)
    if resized.dtype != np.float32:
        resized_float = resized.astype(np.float32) / 255.0
    else:
        resized_float = resized
    input_tensor = np.expand_dims(resized_float, axis=0)

    # Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(input_tensor, model)
        overlaid = overlay_gradcam(resized, heatmap)
        gradcam_available = True
    except Exception as e:
        print(f"[!] Grad-CAM failed: {e}")
        overlaid = resized
        heatmap  = np.zeros(img_size)
        gradcam_available = False

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0f1117")

    titles    = ["Original X-Ray", "Grad-CAM Heatmap", "Overlay"]
    images    = [resized, heatmap, overlaid]
    cmaps_list= ["gray", "jet", None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps_list):
        ax.set_facecolor("#0f1117")
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")

    status = f"{prediction_label}  ({confidence:.1f}%)"
    color  = "#e74c3c" if "Abnormal" in prediction_label else "#2ecc71"
    fig.suptitle(status, color=color, fontsize=13, fontweight="bold", y=1.01)

    if not gradcam_available:
        axes[1].text(0.5, 0.5, "Grad-CAM\nnot available\nfor this model",
                     ha="center", va="center", color="gray", transform=axes[1].transAxes)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gradcam.py <model_path.keras> <image_path.jpg>")
        sys.exit(1)

    model = tf.keras.models.load_model(sys.argv[1])
    img   = cv2.imread(sys.argv[2])
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = generate_gradcam_figure(img, model, prediction_label="Test", confidence=85.0)
    fig.savefig("plots/gradcam_test.png", dpi=150, bbox_inches="tight")
    print("[✔] Grad-CAM saved → plots/gradcam_test.png")
