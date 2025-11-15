import tensorflow as tf
import numpy as np
import cv2

def find_last_conv_layer(model):
    """
    Finds the last convolutional layer in the given model.
    """
    for layer in reversed(model.layers):
        if len(getattr(layer, "output_shape", [])) == 4:
            return layer.name
    raise ValueError("No 4D convolutional layer found in the model.")


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generates Grad-CAM heatmap for the given model and image.

    Args:
        model: Trained CNN model.
        img_array: Input image as numpy array of shape (1, H, W, 3) scaled between 0-1.
        last_conv_layer_name: Optional string; layer name to use for Grad-CAM.

    Returns:
        heatmap: 2D numpy array (H, W) scaled 0–1.
    """
    # Automatically detect last conv layer if not specified
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Create a model that maps the input image to activations + predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # Compute gradients of the top predicted class
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled = pooled_grads.numpy()

    # Weight the convolution outputs by the importance of each channel
    for i in range(pooled.shape[-1]):
        conv_outputs[:, :, i] *= pooled[i]

    # Average along the channel dimension to get the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    # Normalize between 0 and 1
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    # Resize to the model input size
    h, w = img_array.shape[1], img_array.shape[2]
    heatmap = cv2.resize(heatmap.astype('float32'), (w, h))
    return heatmap


def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        img: Original image (H, W, 3) RGB format.
        heatmap: 2D heatmap (H, W) scaled 0–1.
        alpha: Transparency level for blending.

    Returns:
        overlay: Combined RGB image (H, W, 3).
    """
    # Convert heatmap to color
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize to match input image
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

    # Blend heatmap with original image
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay
