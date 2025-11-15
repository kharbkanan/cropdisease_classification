# Quick Start Guide - Plant Disease Classification

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
```bash
python reorganize_for_disease_training.py
```
*This step reorganizes your dataset into the correct structure for training.*

### Step 3: Train Model
```bash
python train_disease_model.py
```
*Training takes 30 minutes to several hours depending on your hardware.*

### Step 4: Run Web App
```bash
streamlit run app.py
```
*The app will open at http://localhost:8501*

## 📋 Prerequisites

- Python 3.8+
- TensorFlow 2.18+
- Streamlit 1.28+
- Dataset with plant disease images

## 🎯 Quick Test

Test the prediction system:
```bash
python test_prediction.py
```

## 📁 Important Files

- `train_disease_model.py` - Train the disease classification model
- `app.py` - Streamlit web application
- `predict_disease.py` - Prediction module
- `remedies.json` - Disease treatment information
- `models/` - Trained models (created after training)

## 🔧 Configuration

Edit `train_disease_model.py` to adjust:
- `EPOCHS` - Number of training epochs (default: 10)
- `BATCH_SIZE` - Training batch size (default: 32)
- `IMG_SIZE` - Image input size (default: 224x224)

## 📊 Supported Crops

- 🌽 Corn (4 diseases)
- 🥔 Potato (3 diseases)
- 🌾 Rice (4 diseases)
- 🎋 Sugarcane (3 diseases)
- 🌾 Wheat (3 diseases)

**Total: 17 disease classes**

## 💡 Tips

1. **Clear Images**: Use well-lit, high-resolution images
2. **Focus on Leaves**: Ensure the leaf area is clearly visible
3. **Good Quality**: Avoid blurry or dark images
4. **Supported Formats**: JPG, PNG, JPEG

## 🐛 Troubleshooting

### Model Not Found
- Run `train_disease_model.py` first
- Check that `models/disease_classifier.keras` exists

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Prediction Errors
- Ensure model is trained on the correct dataset structure
- Verify image preprocessing (224x224, RGB)

## 📚 More Information

See `instructions.txt` for detailed documentation and `README.md` for project overview.

## 🎓 Next Steps

1. Train your model with your dataset
2. Test predictions with `test_prediction.py`
3. Launch web app with `streamlit run app.py`
4. Upload images and get disease predictions!

---

**Happy Plant Disease Detection! 🌿🔍**

