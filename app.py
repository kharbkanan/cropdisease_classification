"""
Streamlit Web App for Plant Disease Classification
Interactive web interface for uploading images and predicting plant diseases
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from predict_disease import DiseasePredictor, get_predictor

# === Page Configuration ===
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1B5E20;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 30px;
        background-color: #C8E6C9;
        border-radius: 5px;
        display: flex;
        align-items: center;
        padding: 0 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# === Initialize Predictor ===
@st.cache_resource
def load_predictor():
    """Load the disease predictor (cached)"""
    try:
        predictor = DiseasePredictor()
        return predictor, None
    except Exception as e:
        return None, str(e)

predictor, error = load_predictor()

# === Header ===
st.markdown('<h1 class="main-header">🌿 Plant Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a leaf image to identify the crop and detect any diseases</p>', unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses deep learning to classify plant diseases from leaf images.
    
    **Supported Crops:**
    - 🌽 Corn
    - 🥔 Potato
    - 🌾 Rice
    - 🎋 Sugarcane
    - 🌾 Wheat
    
    **How to use:**
    1. Upload a clear image of a plant leaf
    2. Click 'Predict Disease'
    3. View predictions and remedies
    """)
    
    st.header("📊 Model Info")
    if predictor:
        st.success(f"✅ Model loaded\n{len(predictor.class_names)} disease classes")
        st.info(f"Available classes:\n{', '.join(predictor.class_names[:5])}...")
    else:
        st.error(f"❌ Model not loaded\n{error}")
    
    st.header("🔗 Instructions")
    st.markdown("""
    - Use clear, well-lit images
    - Focus on the leaf area
    - Ensure good image quality
    - Supported formats: JPG, PNG, JPEG
    """)

# === Main Content ===
if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Please ensure the model files are in the 'models' directory and run the training script first.")
else:
    # === File Upload ===
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        image = Image.open(uploaded_file)
                        
                        # Make prediction
                        results = predictor.predict(image, top_k=3)
                        top_pred = results['top_prediction']
                        all_preds = results['predictions']
                        
                        # Store in session state
                        st.session_state['prediction'] = results
                        st.session_state['image'] = image
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.session_state['prediction'] = None
        
        # Display results if available
        if 'prediction' in st.session_state and st.session_state['prediction']:
            results = st.session_state['prediction']
            top_pred = results['top_prediction']
            
            if top_pred:
                # Main prediction box
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Crop and Disease name
                st.markdown(f'<div class="disease-name">🌾 Crop: {top_pred["crop"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="disease-name">🌿 Disease: {top_pred["display_name"]}</div>', unsafe_allow_html=True)
                
                # Confidence
                confidence = top_pred['confidence']
                st.markdown(f"**Confidence: {confidence:.2f}%**")
                st.progress(confidence / 100.0)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Remedy
                st.markdown("### 💡 Recommended Remedy")
                st.info(top_pred['remedy'])
                
                # Other predictions
                if len(results['predictions']) > 1:
                    st.markdown("### 📊 Other Possible Predictions")
                    for i, pred in enumerate(results['predictions'][1:], 1):
                        with st.expander(f"{i}. {pred['display_name']} ({pred['confidence']:.2f}%)"):
                            st.write(f"**Crop:** {pred['crop']}")
                            st.write(f"**Confidence:** {pred['confidence']:.2f}%")
                            st.write(f"**Remedy:** {pred['remedy']}")
        else:
            st.info("👆 Upload an image and click 'Predict Disease' to see results")
    
    # === Additional Information Section ===
    if 'prediction' in st.session_state and st.session_state['prediction']:
        st.markdown("---")
        st.header("📋 Detailed Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Prediction Details")
            if st.session_state['prediction']['top_prediction']:
                pred = st.session_state['prediction']['top_prediction']
                st.write(f"**Class Name:** {pred['class_name']}")
                st.write(f"**Display Name:** {pred['display_name']}")
                st.write(f"**Crop:** {pred['crop']}")
                st.write(f"**Confidence:** {pred['confidence']:.2f}%")
        
        with col2:
            st.subheader("🌾 Supported Crops")
            crops = ["Corn", "Potato", "Rice", "Sugarcane", "Wheat"]
            for crop in crops:
                st.write(f"• {crop}")
        
        with col3:
            st.subheader("💡 Tips")
            st.write("• Use clear, high-resolution images")
            st.write("• Ensure good lighting")
            st.write("• Focus on the affected leaf area")
            st.write("• Upload images in JPG or PNG format")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem 0;'>"
    "🌿 Plant Disease Classifier | Built with Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)

