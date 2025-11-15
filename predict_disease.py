"""
Plant Disease Prediction Module
Predicts crop and disease from leaf images
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import json
from pathlib import Path

# === Paths ===
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "disease_classifier.keras"
LABEL_MAP_PATH = MODEL_DIR / "disease_label_map.joblib"
REVERSE_MAP_PATH = MODEL_DIR / "disease_reverse_label_map.joblib"
CLASS_NAMES_PATH = MODEL_DIR / "disease_class_names.joblib"
REMEDIES_PATH = Path("remedies.json")

# === Disease name mapping for display ===
DISEASE_DISPLAY_NAMES = {
    # Corn diseases
    "common_rust": "Common Rust",
    "gray_leaf_spot": "Gray Leaf Spot",
    "northern_leaf_blight": "Northern Leaf Blight",
    "healthy": "Healthy",
    
    # Potato diseases
    "early_blight": "Early Blight",
    "late_blight": "Late Blight",
    
    # Rice diseases
    "brown_spot": "Brown Spot",
    "leaf_blast": "Leaf Blight",
    "neck_blast": "Neck Blast",
    
    # Sugarcane diseases
    "bacterial_blight": "Bacterial Blight",
    "red_rot": "Red Rot",
    
    # Wheat diseases
    "brown_rust": "Brown Rust",
    "yellow_rust": "Yellow Rust",
}

# Crop mapping from disease folder names
CROP_MAPPING = {
    "common_rust": "Corn",
    "gray_leaf_spot": "Corn",
    "northern_leaf_blight": "Corn",
    "corn": "Corn",
    
    "early_blight": "Potato",
    "late_blight": "Potato",
    "potato": "Potato",
    
    "brown_spot": "Rice",
    "leaf_blast": "Rice",
    "neck_blast": "Rice",
    "rice": "Rice",
    
    "bacterial_blight": "Sugarcane",
    "red_rot": "Sugarcane",
    "sugarcane": "Sugarcane",
    
    "brown_rust": "Wheat",
    "yellow_rust": "Wheat",
    "wheat": "Wheat",
}

class DiseasePredictor:
    def __init__(self):
        """Initialize the disease predictor"""
        self.model = None
        self.class_indices = None
        self.reverse_label_map = None
        self.class_names = None
        self.remedies = {}
        self.load_model()
        self.load_remedies()
    
    def load_model(self):
        """Load the trained model and label mappings"""
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
            print(f"📂 Loading model from {MODEL_PATH}...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully")
            
            # Load label maps
            if LABEL_MAP_PATH.exists():
                self.class_indices = joblib.load(LABEL_MAP_PATH)
                print("✅ Label map loaded")
            
            if REVERSE_MAP_PATH.exists():
                self.reverse_label_map = joblib.load(REVERSE_MAP_PATH)
            else:
                # Create reverse map from class_indices
                self.reverse_label_map = {v: k for k, v in self.class_indices.items()}
            
            if CLASS_NAMES_PATH.exists():
                self.class_names = joblib.load(CLASS_NAMES_PATH)
            else:
                self.class_names = sorted(self.class_indices.keys())
            
            print(f"✅ Loaded {len(self.class_names)} disease classes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_remedies(self):
        """Load remedies from JSON file"""
        try:
            if REMEDIES_PATH.exists():
                with open(REMEDIES_PATH, 'r', encoding='utf-8') as f:
                    self.remedies = json.load(f)
                print(f"✅ Loaded {len(self.remedies)} remedies")
            else:
                print("⚠️ Remedies file not found, using defaults")
                self.remedies = {}
        except Exception as e:
            print(f"⚠️ Error loading remedies: {e}")
            self.remedies = {}
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image array
        """
        if isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            img = Image.fromarray(image).convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image, top_k=3):
        """
        Predict disease from image
        
        Args:
            image: PIL Image or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        pred_probs = predictions[0]
        
        # Get top k predictions
        top_indices = np.argsort(pred_probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.reverse_label_map[idx]
            confidence = float(pred_probs[idx] * 100)
            
            # Get crop name
            crop = self.get_crop_name(class_name)
            
            # Get display name
            display_name = self.get_display_name(class_name)
            
            # Get remedy
            remedy_key = self.get_remedy_key(class_name, crop)
            remedy = self.remedies.get(remedy_key, "Consult with an agricultural expert for treatment recommendations.")
            
            results.append({
                'class_name': class_name,
                'display_name': display_name,
                'crop': crop,
                'confidence': confidence,
                'remedy': remedy
            })
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }
    
    def get_crop_name(self, disease_name):
        """Extract crop name from disease name"""
        # Disease name format: "crop_disease" (e.g., "corn_common_rust")
        disease_lower = disease_name.lower()
        
        # Extract crop from disease name
        if disease_lower.startswith('corn_'):
            return "Corn"
        elif disease_lower.startswith('potato_'):
            return "Potato"
        elif disease_lower.startswith('rice_'):
            return "Rice"
        elif disease_lower.startswith('sugarcane_'):
            return "Sugarcane"
        elif disease_lower.startswith('wheat_'):
            return "Wheat"
        
        # Check crop mapping for backward compatibility
        for key, crop in CROP_MAPPING.items():
            if key in disease_lower:
                return crop
        
        return "Unknown"
    
    def get_display_name(self, disease_name):
        """Get human-readable display name for disease"""
        # Disease name format: "crop_disease" (e.g., "corn_common_rust")
        parts = disease_name.split('_')
        
        if len(parts) >= 2:
            crop = parts[0].capitalize()
            disease_parts = parts[1:]
            disease_key = '_'.join(disease_parts)
            
            # Get display name for disease
            if disease_key in DISEASE_DISPLAY_NAMES:
                disease_display = DISEASE_DISPLAY_NAMES[disease_key]
            else:
                # Format: "common rust" -> "Common Rust"
                disease_display = ' '.join(word.capitalize() for word in disease_key.split('_'))
            
            # Combine crop and disease
            return f"{crop} - {disease_display}"
        else:
            # Fallback: capitalize and format
            return disease_name.replace('_', ' ').title()
    
    def get_remedy_key(self, disease_name, crop):
        """Get the key for remedies dictionary"""
        # Try different key formats
        keys_to_try = [
            disease_name,  # Original format
            f"{crop.lower()}_{disease_name}",  # With crop prefix
            disease_name.replace('_', ' '),  # With spaces
        ]
        
        for key in keys_to_try:
            if key in self.remedies:
                return key
        
        # Return the original disease name as fallback
        return disease_name


# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor

def predict_from_image(image):
    """Convenience function to predict from image"""
    predictor = get_predictor()
    return predictor.predict(image)

if __name__ == "__main__":
    # Test the predictor
    print("Testing Disease Predictor...")
    predictor = DiseasePredictor()
    print("✅ Predictor initialized successfully")
    print(f"   Available classes: {len(predictor.class_names)}")
    print(f"   Sample classes: {predictor.class_names[:5]}")

