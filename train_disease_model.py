"""
Train a disease classification model on all disease classes from data/train
This model will classify 17 disease classes directly.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === Paths ===
# Use reorganized disease-level dataset
BASE_DIR = Path("data/train_diseases")
# Fallback to original if reorganized doesn't exist
if not BASE_DIR.exists():
    BASE_DIR = Path("data/train_diseases")
    print("⚠️ Using data/train (crop-level). Consider running reorganize_for_disease_training.py first for disease-level classification.")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

print("=" * 60)
print("🌾 Plant Disease Classification Model Training")
print("=" * 60)

# === Data Augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# === Training and Validation Generators ===
print("\n📂 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("\n📂 Loading validation data...")
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class names and indices
class_names = sorted(train_generator.class_indices.keys())
class_indices = train_generator.class_indices
num_classes = len(class_names)

print(f"\n✅ Found {num_classes} disease classes:")
for i, class_name in enumerate(class_names):
    print(f"   {i+1}. {class_name}")

# === Build Model ===
print("\n🏗️  Building model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model initially
base_model.trainable = False

# Add custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model built with {model.count_params():,} parameters")

# === Callbacks ===
checkpoint_path = MODEL_DIR / "disease_classifier.keras"
callbacks = [
    ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# === Train Model ===
print("\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    shuffle=False,
    verbose=1
)

# === Fine-tuning: Unfreeze some layers ===
print("\n🔄 Fine-tuning model...")
base_model.trainable = True
# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# === Save Final Model and Label Map ===
print("\n💾 Saving model and label mappings...")
model.save(checkpoint_path)
print(f"✅ Model saved to: {checkpoint_path}")

# Save class indices mapping
label_map_path = MODEL_DIR / "disease_label_map.joblib"
joblib.dump(class_indices, label_map_path)
print(f"✅ Label map saved to: {label_map_path}")

# Save reverse mapping (index -> class name)
reverse_label_map = {v: k for k, v in class_indices.items()}
reverse_map_path = MODEL_DIR / "disease_reverse_label_map.joblib"
joblib.dump(reverse_label_map, reverse_map_path)
print(f"✅ Reverse label map saved to: {reverse_map_path}")

# Save class names list
class_names_path = MODEL_DIR / "disease_class_names.joblib"
joblib.dump(class_names, class_names_path)
print(f"✅ Class names saved to: {class_names_path}")

print("\n" + "=" * 60)
print("✅ Training completed successfully!")
print("=" * 60)
print(f"\n📊 Final Results:")
print(f"   Training accuracy: {max(history.history['accuracy']):.4f}")
print(f"   Validation accuracy: {max(history.history['val_accuracy']):.4f}")
if 'history_finetune' in locals():
    print(f"   Fine-tuned validation accuracy: {max(history_finetune.history['val_accuracy']):.4f}")

