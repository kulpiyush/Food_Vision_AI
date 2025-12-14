"""
Train YOLO Model on Indian Food Dataset
Step 2.2: Fine-tune YOLOv8 on the SOHL Multi-Dish Indian Food Dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import os

def train_yolo_model(
    data_yaml="data/yolo_training_data/dataset.yaml",
    model_name="yolov8s",  # Changed from yolov8n to yolov8s for better accuracy
    epochs=200,  # Increased from 100 to 200
    batch_size=16,  # Increased from 8 to 16 (if GPU memory allows)
    img_size=640,
    output_dir="models/weights"
):
    """
    Train YOLO model on Indian food dataset
    
    Args:
        data_yaml: Path to dataset.yaml file
        model_name: YOLO model name (yolov8n, yolov8s, yolov8m, etc.)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        output_dir: Directory to save trained model
    """
    data_path = Path(data_yaml)
    
    if not data_path.exists():
        print(f"❌ Dataset YAML not found: {data_yaml}")
        print("\nPlease prepare the dataset first:")
        print("  python scripts/prepare_yolo_dataset.py prepare")
        return None
    
    print("=" * 60)
    print("Training YOLO Model on Indian Food Dataset")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained YOLO model
    print(f"\n📥 Loading pretrained {model_name}...")
    model = YOLO(f"{model_name}.pt")
    
    # Train the model
    print(f"\n🚀 Starting training...")
    print("This may take a while (3-6 hours on CPU, 30-60 min on GPU)...")
    
    try:
        results = model.train(
            data=str(data_path.absolute()),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            # Augmentation parameters for better accuracy
            hsv_h=0.02,          # Hue augmentation (slightly increased)
            hsv_s=0.7,           # Saturation augmentation
            hsv_v=0.4,           # Value augmentation
            degrees=10.0,        # Rotation ±10 degrees (was 0.0)
            translate=0.1,       # Translation
            scale=0.5,           # Scaling
            shear=5.0,           # Shearing ±5 degrees (was 0.0)
            perspective=0.0,     # Perspective transform
            flipud=0.0,          # Vertical flip
            fliplr=0.5,          # Horizontal flip
            mosaic=1.0,          # Mosaic augmentation
            mixup=0.1,           # Mixup augmentation (was 0.0)
            copy_paste=0.0,      # Copy-paste augmentation
            # Training parameters
            patience=50,         # Early stopping patience
            save_period=10,      # Save checkpoint every 10 epochs
            optimizer='AdamW',   # Use AdamW optimizer (better than SGD for small datasets)
            lr0=0.001,           # Initial learning rate (lower for fine-tuning)
            lrf=0.1,             # Final learning rate factor
            momentum=0.937,      # SGD momentum (if using SGD)
            weight_decay=0.0005, # L2 regularization
            warmup_epochs=3.0,   # Warmup epochs
            warmup_momentum=0.8, # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias LR
            # Output settings
            project=output_dir,
            name="food_detector",
            save=True,
            plots=True,
            val=True,
            verbose=True
        )
        
        # Model will be saved automatically to:
        # {output_dir}/food_detector/weights/best.pt
        
        best_model_path = Path(output_dir) / "food_detector" / "weights" / "best.pt"
        
        # Also copy to standard location
        final_model_path = Path(output_dir) / "food_detector_yolo.pt"
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"\n✅ Model saved to: {final_model_path}")
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best model: {best_model_path}")
        print(f"Final model: {final_model_path}")
        
        if results:
            print(f"\n📊 Training Results:")
            print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        print("\n✅ Model is ready to use!")
        print(f"   Update app.py to use: {final_model_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model on Indian food dataset")
    parser.add_argument("--data", type=str, default="data/yolo_training_data/dataset.yaml",
                       help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, default="yolov8s",
                       help="YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--output", type=str, default="models/weights",
                       help="Output directory for trained model")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.data).exists():
        print(f"❌ Dataset not found at {args.data}")
        print("\nPlease prepare the dataset first:")
        print("  python scripts/prepare_yolo_dataset.py prepare")
        exit(1)
    
    # Train
    train_yolo_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        output_dir=args.output
    )

