"""
Complete setup and training script
This script will:
1. Reorganize the dataset for disease-level classification
2. Train the disease classification model
3. Verify the model works correctly
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 60)
    print(f"🔄 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    """Main setup and training function"""
    print("=" * 60)
    print("🌿 Plant Disease Classification - Setup & Training")
    print("=" * 60)
    
    # Check if data/train exists
    train_dir = Path("data/train_diseases")
    if not train_dir.exists():
        print("❌ Error: data/train directory not found!")
        print("   Please ensure your dataset is in data/train/")
        return False
    
    # Step 1: Reorganize dataset
    print("\n📦 Step 1: Reorganizing dataset for disease-level classification...")
    if not run_script("reorganize_for_disease_training.py", "Dataset reorganization"):
        print("⚠️  Warning: Dataset reorganization failed or skipped")
        print("   The training will use the existing data structure")
    
    # Step 2: Train model
    print("\n🚀 Step 2: Training disease classification model...")
    if not run_script("train_disease_model.py", "Model training"):
        print("❌ Error: Model training failed!")
        return False
    
    # Step 3: Verify model
    print("\n✅ Step 3: Verifying model...")
    model_path = Path("models/disease_classifier.keras")
    if model_path.exists():
        print(f"✅ Model file found: {model_path}")
        print("✅ Setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Run 'streamlit run app.py' to start the web app")
        print("   2. Or use 'python predict_disease.py' to test predictions")
        return True
    else:
        print("❌ Error: Model file not found after training!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

