import io
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from IPython.display import display
import ipywidgets as widgets

# === Load Model ===
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crop_classifier.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    raise SystemExit("Please check your model path and try again.")

# === Load Remedies ===
try:
    with open("remedies.json", "r") as f:
        remedies = json.load(f)
    print("✅ Remedies loaded successfully.")
except FileNotFoundError:
    remedies = {}
    print("⚠️ remedies.json not found — please create one like:")
    print("{'corn_common_rust': 'Use fungicide spray...'}")

# === Class Labels (adjust based on your retrained model) ===
classes = [
    "corn_common_rust",
    "corn_gray_leaf_spot",
    "corn_healthy",
    "corn_northern_leaf_blight",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "rice_brown_spot",
    "rice_healthy",
    "rice_leaf_blast",
    "rice_neck_blast",
    "sugarcane_bacterial_blight",
    "sugarcane_healthy",
    "sugarcane_red_rot",
    "wheat_brown_rust",
    "wheat_healthy",
    "wheat_yellow_rust"
]

# === Widgets for Jupyter ===
uploader = widgets.FileUpload(accept='image/*', multiple=False)
predict_btn = widgets.Button(description="🔍 Predict Disease", button_style='success')
output = widgets.Output()

display(widgets.VBox([
    widgets.HTML("<h3>📷 Upload a leaf image, then click 'Predict Disease'</h3>"),
    uploader,
    predict_btn,
    output
]))

# === Prediction Function ===
def process_upload(change=None):
    with output:
        output.clear_output()
        if not uploader.value:
            print("⚠️ Please upload an image first.")
            return

        # --- Fix uploader handling ---
        upload_data = uploader.value

        # Handle different formats (tuple or dict)
        if isinstance(upload_data, dict):
            v = next(iter(upload_data.values()))
        elif isinstance(upload_data, tuple) and len(upload_data) > 0 and isinstance(upload_data[0], dict):
            v = upload_data[0]
        else:
            print("⚠️ Could not read uploaded image.")
            return

        img_bytes = v['content']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        display(img)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        confidence = np.max(pred) * 100
        disease_name = classes[pred_class]
        remedy = remedies.get(disease_name, "No remedy found for this disease.")

        # Display results
        print("\n✅ Prediction Complete!")
        print(f"🌿 Predicted Disease: {disease_name}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print(f"💡 Suggested Remedy: {remedy}")

# === Link button click to function ===
predict_btn.on_click(process_upload)

print("📷 Step 1: Upload a leaf image\n🔍 Step 2: Click 'Predict Disease'")
