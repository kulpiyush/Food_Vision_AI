#!/usr/bin/env python3
"""
Quick script to verify the model is correctly loaded
"""

import os
from pathlib import Path

def verify_model():
    model_path = Path("models/weights/food_detector_yolo.pt")
    
    print("=" * 60)
    print("Model Verification")
    print("=" * 60)
    
    # Check if file exists
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print(f"\n📥 Please download the new model from server:")
        print(f"   Location: models/weights/food_detector3/weights/best.pt")
        print(f"   Save as: {model_path}")
        return False
    
    # Check file size
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Model file found: {model_path}")
    print(f"   Size: {file_size_mb:.1f} MB")
    
    # Expected size for yolov8s: ~22.5 MB
    if file_size_mb < 20:
        print(f"⚠️  Warning: Model size is {file_size_mb:.1f} MB")
        print(f"   Expected ~22.5 MB for yolov8s")
        print(f"   You might still be using the old yolov8n model (6.0 MB)")
    else:
        print(f"✅ Model size looks correct for yolov8s (~22.5 MB)")
    
    # Try loading the model
    try:
        from ultralytics import YOLO
        print(f"\n🔄 Loading model...")
        model = YOLO(str(model_path))
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {len(model.names)}")
        print(f"   Class names: {list(model.names.values())[:5]}...")
        
        # Check if it's yolov8s by parameter count
        # yolov8s has ~11M parameters, yolov8n has ~3M
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"   Parameters: {total_params:,}")
        
        if total_params > 10_000_000:
            print(f"✅ This appears to be yolov8s (11M+ parameters)")
        elif total_params < 5_000_000:
            print(f"⚠️  This appears to be yolov8n (3M parameters)")
            print(f"   You should replace it with the new yolov8s model")
        else:
            print(f"✅ Model architecture looks correct")
        
        print(f"\n🎉 Model is ready to use!")
        print(f"   Run: streamlit run app.py")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    verify_model()

