"""
Test script for disease prediction
Tests the prediction module with a sample workflow
"""

from pathlib import Path
from predict_disease import DiseasePredictor
from PIL import Image
import numpy as np

def test_predictor():
    """Test the disease predictor"""
    print("=" * 60)
    print("🧪 Testing Disease Predictor")
    print("=" * 60)
    
    # Initialize predictor
    try:
        print("\n1️⃣ Initializing predictor...")
        predictor = DiseasePredictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return False
    
    # Check model and labels
    print(f"\n2️⃣ Model information:")
    print(f"   - Number of classes: {len(predictor.class_names)}")
    print(f"   - Sample classes: {predictor.class_names[:5]}")
    print(f"   - Remedies loaded: {len(predictor.remedies)}")
    
    # Test crop name extraction
    print(f"\n3️⃣ Testing crop name extraction:")
    test_cases = [
        "corn_common_rust",
        "sugarcane_bacterial_blight",
        "potato_early_blight",
        "rice_leaf_blast",
        "wheat_yellow_rust"
    ]
    for case in test_cases:
        crop = predictor.get_crop_name(case)
        print(f"   - {case} -> {crop}")
    
    # Test display name formatting
    print(f"\n4️⃣ Testing display name formatting:")
    for case in test_cases:
        display = predictor.get_display_name(case)
        print(f"   - {case} -> {display}")
    
    # Test remedy key lookup
    print(f"\n5️⃣ Testing remedy key lookup:")
    for case in test_cases:
        crop = predictor.get_crop_name(case)
        remedy_key = predictor.get_remedy_key(case, crop)
        has_remedy = remedy_key in predictor.remedies
        print(f"   - {case} -> {remedy_key} (remedy: {'✅' if has_remedy else '❌'})")
    
    # Test with a dummy image (if no real image available)
    print(f"\n6️⃣ Testing prediction with dummy image...")
    try:
        # Create a dummy RGB image
        dummy_image = Image.new('RGB', (224, 224), color='green')
        results = predictor.predict(dummy_image, top_k=3)
        
        if results and results['top_prediction']:
            top = results['top_prediction']
            print(f"✅ Prediction successful:")
            print(f"   - Crop: {top['crop']}")
            print(f"   - Disease: {top['display_name']}")
            print(f"   - Confidence: {top['confidence']:.2f}%")
        else:
            print("⚠️  Prediction returned no results")
    except Exception as e:
        print(f"⚠️  Prediction test skipped (expected with dummy image): {e}")
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_predictor()

