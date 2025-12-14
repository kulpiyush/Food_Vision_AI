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
    model_name="yolov8n",
    epochs=100,
    batch_size=8,
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
            project=output_dir,
            name="food_detector",
            save=True,
            plots=True,
            val=True
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
    parser.add_argument("--model", type=str, default="yolov8n",
                       help="YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
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

