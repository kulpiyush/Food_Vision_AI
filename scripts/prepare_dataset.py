"""
Dataset Preparation Script
Step 2.1: Organize and prepare training dataset for fine-tuning

This script helps organize images into the required folder structure
for fine-tuning the vision model.
"""

import os
import shutil
from pathlib import Path
from typing import List
import random

# Indian food classes (matching vision_model.py)
INDIAN_FOOD_CLASSES = [
    "Biryani", "Dosa", "Idli", "Samosa", "Curry",
    "Naan", "Roti", "Dal", "Paneer Tikka", "Butter Chicken",
    "Palak Paneer", "Chole", "Rajma", "Aloo Gobi", "Baingan Bharta"
]


def create_dataset_structure(base_path="data/training_data", train_split=0.8):
    """
    Create the dataset folder structure
    
    Args:
        base_path (str): Base path for training data
        train_split (float): Fraction of data for training (rest for validation)
    """
    base = Path(base_path)
    
    # Create directories
    train_dir = base / "train"
    val_dir = base / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each food class
    for food_class in INDIAN_FOOD_CLASSES:
        (train_dir / food_class).mkdir(parents=True, exist_ok=True)
        (val_dir / food_class).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directories for {food_class}")
    
    print(f"\n✅ Dataset structure created at {base_path}")
    print(f"   Train: {train_dir}")
    print(f"   Val: {val_dir}")
    
    return train_dir, val_dir


def organize_images_from_folder(
    source_folder: str,
    base_path: str = "data/training_data",
    train_split: float = 0.8
):
    """
    Organize images from a source folder into train/val structure
    
    Assumes source folder has subfolders named after food classes:
    source_folder/
    ├── biryani/
    │   ├── img1.jpg
    │   └── ...
    ├── dosa/
    └── ...
    
    Args:
        source_folder (str): Path to source folder with images
        base_path (str): Destination base path
        train_split (float): Fraction for training
    """
    source = Path(source_folder)
    train_dir, val_dir = create_dataset_structure(base_path, train_split)
    
    if not source.exists():
        print(f"❌ Source folder not found: {source_folder}")
        return
    
    # Process each food class
    for food_class in INDIAN_FOOD_CLASSES:
        source_class_dir = source / food_class.lower().replace(" ", "_")
        
        # Try different naming conventions
        if not source_class_dir.exists():
            # Try with spaces
            source_class_dir = source / food_class
        if not source_class_dir.exists():
            # Try lowercase
            source_class_dir = source / food_class.lower()
        
        if not source_class_dir.exists():
            print(f"⚠️  No images found for {food_class}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(list(source_class_dir.glob(f"*{ext}")))
        
        if not images:
            print(f"⚠️  No images found for {food_class}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        train_class_dir = train_dir / food_class
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        # Copy to val
        val_class_dir = val_dir / food_class
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
        
        print(f"✅ {food_class}: {len(train_images)} train, {len(val_images)} val")


def verify_dataset(base_path="data/training_data"):
    """
    Verify dataset structure and count images
    
    Args:
        base_path (str): Base path to dataset
    """
    base = Path(base_path)
    train_dir = base / "train"
    val_dir = base / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Dataset structure not found. Run create_dataset_structure() first.")
        return
    
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    total_train = 0
    total_val = 0
    
    for food_class in INDIAN_FOOD_CLASSES:
        train_class_dir = train_dir / food_class
        val_class_dir = val_dir / food_class
        
        train_count = len(list(train_class_dir.glob("*.jpg")) + 
                         list(train_class_dir.glob("*.jpeg")) + 
                         list(train_class_dir.glob("*.png")))
        val_count = len(list(val_class_dir.glob("*.jpg")) + 
                       list(val_class_dir.glob("*.jpeg")) + 
                       list(val_class_dir.glob("*.png")))
        
        total_train += train_count
        total_val += val_count
        
        status = "✅" if train_count >= 50 and val_count >= 10 else "⚠️"
        print(f"{status} {food_class:20s} | Train: {train_count:4d} | Val: {val_count:4d}")
    
    print("=" * 60)
    print(f"Total: {total_train} train, {total_val} val images")
    print(f"Total: {total_train + total_val} images")
    
    # Check minimum requirements
    min_per_class = 50
    if total_train < len(INDIAN_FOOD_CLASSES) * min_per_class:
        print(f"\n⚠️  Warning: Recommended minimum is {min_per_class} images per class")
        print(f"   You have ~{total_train // len(INDIAN_FOOD_CLASSES)} images per class on average")
    else:
        print(f"\n✅ Dataset meets minimum requirements!")


def print_dataset_instructions():
    """Print instructions for preparing the dataset"""
    print("\n" + "=" * 60)
    print("Dataset Preparation Instructions")
    print("=" * 60)
    print("""
To prepare your training dataset:

1. **Collect Images:**
   - Minimum: 50-100 images per food category
   - Sources: Kaggle, Food-101, custom photos, etc.
   - Format: JPG, PNG, JPEG

2. **Organize Images:**
   Option A: Place images in folders named after food classes:
   ```
   your_images/
   ├── biryani/
   │   ├── img1.jpg
   │   └── ...
   ├── dosa/
   └── ...
   ```
   Then run: organize_images_from_folder("your_images")

   Option B: Manually place images in:
   ```
   data/training_data/train/biryani/
   data/training_data/val/biryani/
   ```

3. **Verify Dataset:**
   Run: verify_dataset()

4. **Food Classes Needed:**
""")
    for i, food in enumerate(INDIAN_FOOD_CLASSES, 1):
        print(f"   {i:2d}. {food}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            create_dataset_structure()
        elif command == "verify":
            verify_dataset()
        elif command == "organize" and len(sys.argv) > 2:
            organize_images_from_folder(sys.argv[2])
        else:
            print("Usage:")
            print("  python prepare_dataset.py create          # Create folder structure")
            print("  python prepare_dataset.py verify          # Verify dataset")
            print("  python prepare_dataset.py organize <path> # Organize images from folder")
    else:
        print_dataset_instructions()
        print("\nCreating dataset structure...")
        create_dataset_structure()
        print("\nVerifying dataset...")
        verify_dataset()

