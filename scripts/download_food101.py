"""
Download and Extract Food-101 Dataset for Indian Cuisine
Step 2.1: Automated dataset preparation from Food-101

This script helps download Food-101 and extract relevant Indian food categories.
"""

import os
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List
import json

# Our Indian food classes
INDIAN_FOOD_CLASSES = [
    "Biryani", "Dosa", "Idli", "Samosa", "Curry",
    "Naan", "Roti", "Dal", "Paneer Tikka", "Butter Chicken",
    "Palak Paneer", "Chole", "Rajma", "Aloo Gobi", "Baingan Bharta"
]

# Mapping from our class names to Food-101 category names
# Food-101 uses lowercase with underscores
FOOD101_MAPPING = {
    "Samosa": "samosa",  # Direct match
    "Curry": "chicken_curry",  # Food-101 has chicken_curry
    # Note: Food-101 doesn't have all Indian foods, so we'll need to handle this
}

# Food-101 categories that might be useful (even if not exact matches)
FOOD101_SIMILAR = {
    "chicken_curry": "Curry",
    "samosa": "Samosa",
    # We can use these as starting points and supplement with other sources
}


def check_kaggle_available():
    """Check if Kaggle API is available"""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def download_food101_kaggle(output_dir="data/food101_raw"):
    """
    Download Food-101 dataset using Kaggle API
    
    Requires:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - kaggle package: pip install kaggle
    
    Args:
        output_dir (str): Directory to save downloaded files
    """
    if not check_kaggle_available():
        print("❌ Kaggle API not available.")
        print("\nTo use Kaggle download:")
        print("1. Install: pip install kaggle")
        print("2. Get API credentials from: https://www.kaggle.com/account")
        print("3. Save to: ~/.kaggle/kaggle.json")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/dansbecker/food101")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("📥 Downloading Food-101 dataset from Kaggle...")
        print("This may take a while (large dataset ~4.6GB)...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            'dansbecker/food101',
            path=str(output_path),
            unzip=True
        )
        
        print(f"✅ Dataset downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False


def extract_food101_images(source_dir, target_dir="data/training_data", train_split=0.8):
    """
    Extract relevant images from Food-101 dataset
    
    Args:
        source_dir (str): Path to Food-101 dataset
        target_dir (str): Target directory for organized images
        train_split (float): Train/val split ratio
    """
    source = Path(source_dir)
    target = Path(target_dir)
    
    # Food-101 structure:
    # food-101/
    #   ├── images/
    #   │   ├── samosa/
    #   │   ├── chicken_curry/
    #   │   └── ...
    #   ├── meta/
    #   │   ├── train.txt
    #   │   └── test.txt
    
    images_dir = source / "images"
    meta_dir = source / "meta"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        print("Make sure Food-101 is extracted correctly")
        return False
    
    # Read train/test splits from Food-101
    train_file = meta_dir / "train.txt"
    test_file = meta_dir / "test.txt"
    
    train_images = set()
    test_images = set()
    
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_images = {line.strip().split('/')[1] for line in f}
    
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_images = {line.strip().split('/')[1] for line in f}
    
    # Create target structure
    train_target = target / "train"
    val_target = target / "val"
    
    # Process each food category we need
    extracted_count = 0
    
    for our_class, food101_class in FOOD101_MAPPING.items():
        food101_path = images_dir / food101_class
        
        if not food101_path.exists():
            print(f"⚠️  Food-101 category '{food101_class}' not found, skipping {our_class}")
            continue
        
        # Create target directories
        train_class_dir = train_target / our_class
        val_class_dir = val_target / our_class
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(food101_path.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  No images found for {food101_class}")
            continue
        
        # Split based on Food-101's train/test split
        train_count = 0
        val_count = 0
        
        for img_file in image_files:
            img_name = img_file.stem
            
            # Check if in train or test set
            if img_name in train_images:
                shutil.copy2(img_file, train_class_dir / img_file.name)
                train_count += 1
            elif img_name in test_images:
                shutil.copy2(img_file, val_class_dir / img_file.name)
                val_count += 1
            else:
                # If not in either, use our own split
                import random
                if random.random() < train_split:
                    shutil.copy2(img_file, train_class_dir / img_file.name)
                    train_count += 1
                else:
                    shutil.copy2(img_file, val_class_dir / img_file.name)
                    val_count += 1
        
        extracted_count += len(image_files)
        print(f"✅ {our_class}: {train_count} train, {val_count} val images")
    
    print(f"\n✅ Extracted {extracted_count} images total")
    print(f"⚠️  Note: Food-101 only has a few Indian food categories")
    print(f"   You may need to supplement with other sources for:")
    for cls in INDIAN_FOOD_CLASSES:
        if cls not in FOOD101_MAPPING:
            print(f"   - {cls}")
    
    return True


def print_manual_download_instructions():
    """Print instructions for manual download"""
    print("\n" + "=" * 60)
    print("Manual Download Instructions")
    print("=" * 60)
    print("""
Since Food-101 has limited Indian food categories, here's what to do:

1. **Download Food-101:**
   - Visit: https://www.kaggle.com/datasets/dansbecker/food101
   - Click "Download" (requires Kaggle account)
   - Extract the zip file

2. **Extract Relevant Categories:**
   Food-101 has these Indian/related foods:
   - samosa ✅ (direct match)
   - chicken_curry ✅ (can use for "Curry")
   
   Other categories you might find useful:
   - fried_rice (similar to Biryani)
   - naan (might be available in some versions)

3. **Run This Script:**
   python scripts/download_food101.py extract /path/to/food-101

4. **Supplement with Other Sources:**
   For foods not in Food-101, you can:
   - Search Kaggle for "Indian food dataset"
   - Use Google Images (with proper attribution)
   - Collect custom photos
   - Use other food datasets

5. **Alternative: Use Multiple Datasets**
   - Food-101 for samosa, curry
   - Indian Food Images dataset from Kaggle
   - Custom collection for others
""")
    print("=" * 60)


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Food-101 Dataset Downloader")
        print("=" * 60)
        print("\nUsage:")
        print("  python download_food101.py download          # Download using Kaggle API")
        print("  python download_food101.py extract <path>    # Extract from existing download")
        print("  python download_food101.py manual            # Show manual instructions")
        print("\nNote: Food-101 has limited Indian food categories.")
        print("You may need to supplement with other sources.")
        return
    
    command = sys.argv[1]
    
    if command == "download":
        success = download_food101_kaggle()
        if success:
            print("\n✅ Download complete!")
            print("Next: Run 'python download_food101.py extract data/food101_raw'")
    elif command == "extract":
        if len(sys.argv) < 3:
            print("❌ Please provide path to Food-101 dataset")
            print("Usage: python download_food101.py extract /path/to/food-101")
            return
        source_path = sys.argv[2]
        extract_food101_images(source_path)
    elif command == "manual":
        print_manual_download_instructions()
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()

