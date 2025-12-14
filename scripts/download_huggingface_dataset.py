"""
Download and Convert HuggingFace Indian Food Dataset
Converts YOLO format to classification format for our model

Dataset: SohlHealth/sohl-multidish-yolo-dataset
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple
import random

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Our Indian food classes
OUR_CLASSES = [
    "Biryani", "Dosa", "Idli", "Samosa", "Curry",
    "Naan", "Roti", "Dal", "Paneer Tikka", "Butter Chicken",
    "Palak Paneer", "Chole", "Rajma", "Aloo Gobi", "Baingan Bharta"
]

# Mapping from dataset classes to our classes
# The dataset has broader categories, we'll map them
CLASS_MAPPING = {
    # Direct or close matches
    "bread_or_Roti_naan": ["Naan", "Roti"],  # Can be either
    "curry_dish": ["Curry", "Butter Chicken", "Palak Paneer"],  # General curry
    "rice_dish": ["Biryani"],  # Rice dishes
    "snack_item": ["Samosa"],  # Snacks
    "Dal_or_sambar": ["Dal"],  # Direct match
    "south_indian_breakfast": ["Dosa", "Idli"],  # South Indian breakfast
    "dry_vegetable": ["Aloo Gobi", "Baingan Bharta"],  # Dry vegetables
    # Note: Some classes like Chole, Rajma, Paneer Tikka might not map directly
    # We'll need to supplement or use similar categories
}

# Reverse mapping for easier lookup
REVERSE_MAPPING = {}
for dataset_class, our_classes in CLASS_MAPPING.items():
    for our_class in our_classes:
        if our_class not in REVERSE_MAPPING:
            REVERSE_MAPPING[our_class] = []
        REVERSE_MAPPING[our_class].append(dataset_class)


def download_huggingface_dataset(repo_id="SohlHealth/sohl-multidish-yolo-dataset", output_dir="data/hf_dataset"):
    """
    Download dataset from HuggingFace
    
    Args:
        repo_id (str): HuggingFace dataset repository ID
        output_dir (str): Directory to save downloaded dataset
    
    Returns:
        Path to downloaded dataset
    """
    if not HF_AVAILABLE:
        print("❌ HuggingFace Hub not available.")
        print("\nInstall it with:")
        print("  pip install huggingface_hub")
        return None
    
    print(f"📥 Downloading dataset from HuggingFace: {repo_id}")
    print("This may take a while...")
    
    try:
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir
        )
        print(f"✅ Dataset downloaded to: {dataset_path}")
        return Path(dataset_path)
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def parse_yolo_annotation(annotation_file: Path) -> List[Tuple[int, List[float]]]:
    """
    Parse YOLO format annotation file
    
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    
    Returns:
        List of (class_id, bbox) tuples
    """
    annotations = []
    if not annotation_file.exists():
        return annotations
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]  # center_x, center_y, width, height
                annotations.append((class_id, bbox))
    
    return annotations


def crop_image_from_bbox(image: Image.Image, bbox: List[float]) -> Image.Image:
    """
    Crop image using YOLO bbox format
    
    Args:
        image: PIL Image
        bbox: [center_x, center_y, width, height] normalized 0-1
    
    Returns:
        Cropped PIL Image
    """
    img_width, img_height = image.size
    
    # Convert normalized bbox to pixel coordinates
    center_x = bbox[0] * img_width
    center_y = bbox[1] * img_height
    width = bbox[2] * img_width
    height = bbox[3] * img_height
    
    # Calculate crop box (left, top, right, bottom)
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(img_width, int(center_x + width / 2))
    bottom = min(img_height, int(center_y + height / 2))
    
    # Crop with some padding
    padding = 10
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img_width, right + padding)
    bottom = min(img_height, bottom + padding)
    
    if right > left and bottom > top:
        return image.crop((left, top, right, bottom))
    return image


def convert_yolo_to_classification(
    dataset_path: Path,
    output_dir: str = "data/training_data",
    train_split: float = 0.8,
    min_crop_size: int = 64
):
    """
    Convert YOLO format dataset to classification format
    
    Args:
        dataset_path: Path to downloaded HuggingFace dataset
        output_dir: Output directory for classification dataset
        train_split: Train/val split ratio
        min_crop_size: Minimum size for cropped images
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Dataset structure not found.")
        print(f"   Expected: {images_dir} and {labels_dir}")
        return False
    
    # Read class names from dataset.yaml if available
    dataset_yaml = dataset_path / "dataset.yaml"
    class_names = None
    
    if dataset_yaml.exists():
        import yaml
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
            class_names = config.get('names', [])
            print(f"✅ Found {len(class_names)} classes in dataset")
    
    if not class_names:
        # Use default class names from dataset description
        class_names = [
            "bread_or_Roti_naan", "curry_dish", "rice_dish", "dry_vegetable",
            "snack_item", "sweet_item", "accompaniment", "Dal_or_sambar",
            "drink", "eggs", "fish_dish", "fruits", "pasta", "salad",
            "soup", "south_indian_breakfast"
        ]
    
    # Create output structure
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    # Create directories for our classes
    for our_class in OUR_CLASSES:
        (train_dir / our_class).mkdir(parents=True, exist_ok=True)
        (val_dir / our_class).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total_cropped = 0
    class_counts = {cls: 0 for cls in OUR_CLASSES}
    
    print(f"\n📦 Processing {len(image_files)} images...")
    
    for img_file in image_files:
        # Find corresponding annotation file
        label_file = labels_dir / (img_file.stem + ".txt")
        
        if not label_file.exists():
            continue
        
        # Load image
        try:
            image = Image.open(img_file).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {img_file}: {e}")
            continue
        
        # Parse annotations
        annotations = parse_yolo_annotation(label_file)
        
        if not annotations:
            continue
        
        # Process each detected food item
        for class_id, bbox in annotations:
            if class_id >= len(class_names):
                continue
            
            dataset_class = class_names[class_id]
            
            # Map to our classes
            target_classes = CLASS_MAPPING.get(dataset_class, [])
            
            if not target_classes:
                # Skip unmapped classes
                continue
            
            # Crop image
            cropped = crop_image_from_bbox(image, bbox)
            
            # Skip if too small
            if cropped.size[0] < min_crop_size or cropped.size[1] < min_crop_size:
                continue
            
            # For each target class, save the cropped image
            for target_class in target_classes:
                # Randomly assign to train or val
                is_train = random.random() < train_split
                target_dir = train_dir if is_train else val_dir
                
                # Save cropped image
                save_path = target_dir / target_class / f"{img_file.stem}_{class_id}_{total_cropped}.jpg"
                cropped.save(save_path, "JPEG", quality=95)
                
                class_counts[target_class] += 1
                total_cropped += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total cropped images: {total_cropped}")
    print("\nImages per class:")
    for cls in OUR_CLASSES:
        count = class_counts[cls]
        status = "✅" if count > 0 else "❌"
        print(f"{status} {cls:20s}: {count:4d} images")
    
    print("\n" + "=" * 60)
    print("⚠️  Note: Some classes may have few/no images")
    print("   You may need to supplement with other sources")
    print("=" * 60)
    
    return True


def main():
    """Main function"""
    import sys
    
    if not HF_AVAILABLE:
        print("Installing huggingface_hub...")
        print("Run: pip install huggingface_hub")
        return
    
    if len(sys.argv) < 2:
        print("HuggingFace Dataset Downloader & Converter")
        print("=" * 60)
        print("\nUsage:")
        print("  python download_huggingface_dataset.py download")
        print("  python download_huggingface_dataset.py convert <dataset_path>")
        print("  python download_huggingface_dataset.py full")
        print("\nDataset: SohlHealth/sohl-multidish-yolo-dataset")
        return
    
    command = sys.argv[1]
    
    if command == "download":
        dataset_path = download_huggingface_dataset()
        if dataset_path:
            print(f"\n✅ Download complete!")
            print(f"Next: Run 'python download_huggingface_dataset.py convert {dataset_path}'")
    elif command == "convert":
        if len(sys.argv) < 3:
            print("❌ Please provide path to dataset")
            print("Usage: python download_huggingface_dataset.py convert <dataset_path>")
            return
        convert_yolo_to_classification(sys.argv[2])
    elif command == "full":
        print("📥 Downloading and converting dataset...")
        dataset_path = download_huggingface_dataset()
        if dataset_path:
            print("\n🔄 Converting to classification format...")
            convert_yolo_to_classification(dataset_path)
            print("\n✅ Complete! Run 'python scripts/prepare_dataset.py verify' to check")
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()

