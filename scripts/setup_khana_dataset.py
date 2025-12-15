#!/usr/bin/env python3
"""
Script to organize Khana dataset into classification format
Khana dataset is classification format (folders by class name)
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import random

PROJECT_ROOT = Path("/export/home/4prasad/piyush/Food_Vision_AI")
DATA_DIR = PROJECT_ROOT / "data"
KHANA_DIR = DATA_DIR / "khana_dataset"
TRAINING_DATA_DIR = DATA_DIR / "training_data"


def analyze_dataset_structure(dataset_path: Path) -> Dict:
    """Analyze the structure of the downloaded dataset"""
    print(f"Analyzing dataset structure in: {dataset_path}")
    
    structure = {
        "has_train_val": False,
        "has_class_folders": False,
        "directories": [],
        "files": [],
        "classification_format": False,
        "actual_data_path": None
    }
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist")
        return structure
    
    # Check if there's a 'khana' subdirectory (common in extracted datasets)
    khana_subdir = dataset_path / "khana"
    if khana_subdir.exists() and khana_subdir.is_dir():
        print(f"Found 'khana' subdirectory, checking inside...")
        dataset_path = khana_subdir
        structure["actual_data_path"] = khana_subdir
    
    # List all items
    items = list(dataset_path.iterdir())
    
    for item in items:
        if item.is_dir():
            structure["directories"].append(item.name)
            # Check if it's already in classification format
            if item.name in ["train", "val", "test"]:
                structure["has_train_val"] = True
                # Check if train has class folders
                train_dir = item
                if train_dir.exists():
                    subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        structure["has_class_folders"] = True
                        structure["classification_format"] = True
            else:
                # Check if this directory contains images (it's a class folder)
                image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png"))
                if image_files:
                    structure["has_class_folders"] = True
        else:
            structure["files"].append(item.name)
    
    if not structure["actual_data_path"]:
        structure["actual_data_path"] = dataset_path
    
    return structure


def get_class_names_from_dataset(dataset_path: Path) -> List[str]:
    """Extract class names from the dataset"""
    class_names = []
    
    # Check if there's a 'khana' subdirectory
    khana_subdir = dataset_path / "khana"
    if khana_subdir.exists() and khana_subdir.is_dir():
        dataset_path = khana_subdir
    
    # Check if labels.txt exists (Khana dataset has this)
    labels_file = dataset_path.parent / "labels.txt" if khana_subdir.exists() else dataset_path.parent / "labels.txt"
    if not labels_file.exists():
        labels_file = dataset_path / "labels.txt"
    
    if labels_file.exists():
        print(f"Found labels.txt, reading class names...")
        with open(labels_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        if class_names:
            return sorted(class_names)
    
    # Check if already split into train/val
    train_dir = dataset_path / "train"
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names
    
    # Check if root has class folders
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if class_dirs:
        # Check if these are class folders (contain images) or other directories
        for class_dir in class_dirs:
            # Check if it contains images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            if image_files:
                class_names.append(class_dir.name)
    
    return sorted(class_names) if class_names else []


def organize_classification_format(source_dir: Path, target_dir: Path, train_split=0.8, val_split=0.1, seed=42):
    """Organize dataset into classification format (train/val/test)"""
    print(f"Organizing dataset from {source_dir} to {target_dir}")
    
    # Check if there's a 'khana' subdirectory
    khana_subdir = source_dir / "khana"
    if khana_subdir.exists() and khana_subdir.is_dir():
        print("Found 'khana' subdirectory, using it as source...")
        source_dir = khana_subdir
    
    # Set random seed
    random.seed(seed)
    
    # Create directory structure
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    test_dir = target_dir / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if source is already in classification format
    structure = analyze_dataset_structure(source_dir)
    
    if structure["classification_format"]:
        print("✅ Dataset is already in classification format. Copying structure...")
        # Copy train/val/test directories
        for split in ["train", "val", "test"]:
            split_dir = source_dir / split
            if split_dir.exists():
                print(f"  Copying {split}/...")
                # Copy each class folder
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        target_class_dir = target_dir / split / class_dir.name
                        shutil.copytree(class_dir, target_class_dir, dirs_exist_ok=True)
        return True
    
    # Check if root has class folders
    class_dirs = [d for d in source_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        print("❌ No class directories found. Expected format:")
        print("   Option 1: train/class1/, train/class2/, ...")
        print("   Option 2: class1/, class2/, ... (will be split)")
        return False
    
    # Check if these are class folders (contain images)
    # Note: Khana dataset images may not have extensions
    valid_class_dirs = []
    for class_dir in class_dirs:
        # Try with extensions first
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPG")) + \
                     list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.PNG"))
        
        # If no files with extensions, check for files without extensions (Khana dataset format)
        if not image_files:
            all_files = [f for f in class_dir.iterdir() if f.is_file()]
            # Filter out non-image files (like .txt, .csv, etc.)
            image_files = [f for f in all_files if not f.suffix.lower() in ['.txt', '.csv', '.json', '.yaml', '.yml']]
        
        if image_files:
            valid_class_dirs.append((class_dir, image_files))
    
    if not valid_class_dirs:
        print("❌ No image files found in class directories")
        print(f"   Checked {len(class_dirs)} directories")
        if class_dirs:
            sample_dir = class_dirs[0]
            sample_files = list(sample_dir.iterdir())[:5]
            print(f"   Sample files in '{sample_dir.name}': {[f.name for f in sample_files]}")
        return False
    
    print(f"✅ Found {len(valid_class_dirs)} classes with images")
    
    # Split each class into train/val/test
    for class_dir, image_files in valid_class_dirs:
        class_name = class_dir.name
        print(f"  Processing {class_name} ({len(image_files)} images)...")
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate splits
        total = len(image_files)
        train_count = int(total * train_split)
        val_count = int(total * val_split)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Create class directories in train/val/test
        for split_name, files, split_dir in [
            ("train", train_files, train_dir),
            ("val", val_files, val_dir),
            ("test", test_files, test_dir)
        ]:
            target_class_dir = split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                shutil.copy2(img_file, target_class_dir / img_file.name)
        
        print(f"    Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return True


def main():
    print("=" * 60)
    print("Khana Dataset Setup Script - Classification Format")
    print("=" * 60)
    
    # Step 1: Analyze the downloaded dataset
    if not KHANA_DIR.exists():
        print(f"Error: Khana dataset directory not found at {KHANA_DIR}")
        print("Please run the download script first:")
        print(f"  ./scripts/download_khana_dataset.sh <FILE_ID>")
        return
    
    print("\nStep 1: Analyzing dataset structure...")
    structure = analyze_dataset_structure(KHANA_DIR)
    print(f"Structure analysis:")
    print(f"  - Has train/val structure: {structure['has_train_val']}")
    print(f"  - Has class folders: {structure['has_class_folders']}")
    print(f"  - Already in classification format: {structure['classification_format']}")
    print(f"  - Directories found: {structure['directories'][:10]}...")  # Show first 10
    
    # Step 2: Get class names
    print("\nStep 2: Extracting class names...")
    class_names = get_class_names_from_dataset(KHANA_DIR)
    if not class_names:
        print("Warning: Could not automatically detect class names.")
        print("Please check the dataset structure.")
        return
    
    print(f"Found {len(class_names)} classes:")
    for i, name in enumerate(class_names[:20]):  # Show first 20
        print(f"  {i}: {name}")
    if len(class_names) > 20:
        print(f"  ... and {len(class_names) - 20} more")
    
    # Step 3: Remove old dataset completely
    print("\nStep 3: Removing old dataset...")
    if TRAINING_DATA_DIR.exists():
        backup_dir = DATA_DIR / f"old_training_data_backup_{Path(__file__).stem}"
        print(f"Backing up old dataset to {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(TRAINING_DATA_DIR, backup_dir)
        shutil.rmtree(TRAINING_DATA_DIR)
        print("Old dataset removed.")
    
    # Step 4: Organize into classification format
    print("\nStep 4: Organizing dataset into classification format...")
    success = organize_classification_format(KHANA_DIR, TRAINING_DATA_DIR)
    
    if not success:
        print("\n❌ Failed to organize dataset automatically.")
        print("Please check the dataset structure and organize manually if needed.")
        return
    
    # Step 5: Save class names
    print("\nStep 5: Saving class names...")
    class_names_file = TRAINING_DATA_DIR / "class_names.txt"
    with open(class_names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✅ Saved class names to {class_names_file}")
    
    # Step 6: Count images
    print("\nStep 6: Dataset statistics...")
    train_count = sum(len(list((TRAINING_DATA_DIR / "train" / cls).glob("*.jpg"))) +
                     len(list((TRAINING_DATA_DIR / "train" / cls).glob("*.jpeg"))) +
                     len(list((TRAINING_DATA_DIR / "train" / cls).glob("*.png")))
                     for cls in class_names if (TRAINING_DATA_DIR / "train" / cls).exists())
    val_count = sum(len(list((TRAINING_DATA_DIR / "val" / cls).glob("*.jpg"))) +
                   len(list((TRAINING_DATA_DIR / "val" / cls).glob("*.jpeg"))) +
                   len(list((TRAINING_DATA_DIR / "val" / cls).glob("*.png")))
                   for cls in class_names if (TRAINING_DATA_DIR / "val" / cls).exists())
    test_count = sum(len(list((TRAINING_DATA_DIR / "test" / cls).glob("*.jpg"))) +
                    len(list((TRAINING_DATA_DIR / "test" / cls).glob("*.jpeg"))) +
                    len(list((TRAINING_DATA_DIR / "test" / cls).glob("*.png")))
                    for cls in class_names if (TRAINING_DATA_DIR / "test" / cls).exists())
    
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"  Test images: {test_count}")
    print(f"  Total images: {train_count + val_count + test_count}")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print(f"Dataset location: {TRAINING_DATA_DIR}")
    print(f"Classes: {len(class_names)}")
    print(f"Class names file: {class_names_file}")
    print("\n🚀 Next Steps:")
    print("1. Verify the dataset structure")
    print("2. Train your model: python scripts/train_classification_model.py")
    print("3. The app will automatically use the new model!")


if __name__ == "__main__":
    main()
