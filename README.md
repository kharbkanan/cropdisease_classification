# 🌿 Plant Disease Classification System

A comprehensive deep learning-based system for classifying plant diseases from leaf images. This project supports multiple crops (Corn, Potato, Rice, Sugarcane, Wheat) and provides an interactive web interface using Streamlit.

## 📋 Features

- **Multi-Crop Support**: Classifies diseases in 5 different crops
- **17 Disease Classes**: Detects various diseases and healthy conditions
- **Interactive Web Interface**: User-friendly Streamlit app for image upload and prediction
- **Detailed Predictions**: Shows crop name, disease name, confidence score, and treatment remedies
- **Top-K Predictions**: Displays multiple possible predictions with confidence scores

## 🎯 Supported Crops and Diseases

### Corn
- Common Rust
- Gray Leaf Spot
- Northern Leaf Blight
- Healthy

### Potato
- Early Blight
- Late Blight
- Healthy

### Rice
- Brown Spot
- Leaf Blast
- Neck Blast
- Healthy

### Sugarcane
- Bacterial Blight
- Red Rot
- Healthy

### Wheat
- Brown Rust
- Yellow Rust
- Healthy

## 🚀 Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📦 Setup and Training

### Step 1: Reorganize Dataset (if needed)

If your dataset is organized by crop folders with disease subfolders, run the reorganization script:

```bash
python reorganize_for_disease_training.py
```

This will create a flat structure: `data/train_diseases/{crop}_{disease}/` for disease-level classification.

### Step 2: Train the Model

Train the disease classification model:

```bash
python train_disease_model.py
```

This will:
- Load images from `data/train_diseases/` (or `data/train/` as fallback)
- Train a MobileNetV2-based model
- Save the model to `models/disease_classifier.keras`
- Save label mappings to `models/` directory

**Training Time**: Depending on your hardware, training may take 30 minutes to several hours.

### Step 3: Verify Model

Test the prediction module:

```bash
python predict_disease.py
```

## 🌐 Running the Web App

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the Web App

1. **Upload Image**: Click "Browse files" and select a plant leaf image
2. **Predict**: Click the "🔍 Predict Disease" button
3. **View Results**: See the predicted crop, disease name, confidence score, and recommended remedy
4. **Explore**: View alternative predictions in the expandable sections

## 📁 Project Structure

```
plant/
├── data/
│   ├── train/              # Original dataset (crop/disease structure)
│   └── train_diseases/     # Reorganized dataset (disease-level)
├── models/                 # Trained models and label mappings
│   ├── disease_classifier.keras
│   ├── disease_label_map.joblib
│   └── disease_reverse_label_map.joblib
├── app.py                  # Streamlit web application
├── predict_disease.py      # Prediction module
├── train_disease_model.py  # Model training script
├── reorganize_for_disease_training.py  # Dataset reorganization script
├── remedies.json           # Disease remedies and treatments
├── requirements.txt        # Python dependencies
└── README.md              # This file
```
📂 How to Use Your Own Dataset
This project allows anyone to train the model with their own images.
Just follow the steps:
🚀 1. Add Your Dataset
Place your dataset inside: plant/data/train/
2.Auto-Generation of Labels & Model
When you run the training file, the project will automatically create:
✔ traindisease.txt
Contains all class names detected from the folder structure.
No need to manually edit anything.
✔ models/ Folder
The trained deep learning model is saved here automatically.
No need to upload your own model
3. Train the Model
Run: python train.py
4. Predict Crop Disease
Once the model is trained, run:
python predict_crop.py --image your_image.jpg

## 🔧 Configuration

### Model Parameters

Edit `train_disease_model.py` to adjust:
- `IMG_SIZE`: Image input size (default: 224x224)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)

### Prediction Parameters

Edit `predict_disease.py` to adjust:
- `top_k`: Number of top predictions to return (default: 3)

## 🐛 Troubleshooting

### Model Not Found Error

If you get a "Model not found" error:
1. Ensure you've run the training script first
2. Check that `models/disease_classifier.keras` exists
3. Verify the model directory path in `predict_disease.py`

### Incorrect Predictions

If predictions are incorrect:
1. Ensure the model was trained on the reorganized dataset
2. Check that the label mappings match your dataset structure
3. Verify that images are preprocessed correctly (224x224, RGB)

### Sugarcane Showing as Corn

This was a known issue that has been fixed by:
1. Training on disease-level classes (not just crop-level)
2. Proper label mapping in the prediction module
3. Correct crop extraction from disease names

## 📊 Model Performance

The model uses:
- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224 RGB images
- **Classes**: 17 disease classes
- **Augmentation**: Rotation, shifts, zoom, flipping, brightness adjustment

## 🔮 Future Improvements

- [ ] Add more crop types
- [ ] Improve model accuracy with more training data
- [ ] Add batch prediction capability
- [ ] Implement GradCAM visualization
- [ ] Add model evaluation metrics
- [ ] Support for video input

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Plant disease images from various sources
- Framework: TensorFlow/Keras for deep learning
- UI: Streamlit for web interface
- Model: MobileNetV2 pre-trained on ImageNet

## 📧 Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

---

**Happy Plant Disease Detection! 🌿🔍**


