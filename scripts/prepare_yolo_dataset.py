"""
Prepare YOLO Dataset for Training
Splits the HuggingFace dataset into train/val for YOLO training
The dataset is already in YOLO format - we just need to organize it!
"""

import os
import shutil
from pathlib import Path
import random
import yaml

def prepare_yolo_dataset(
    source_dir="data/hf_dataset",
    output_dir="data/yolo_training_data",
    train_split=0.8,
    seed=42
):
    """
    Prepare YOLO dataset by splitting into train/val
    
    The dataset is already in YOLO format - we just organize it!
    
    Args:
        source_dir: Source directory with images/ and labels/
        output_dir: Output directory for organized dataset
        train_split: Fraction for training (default 0.8)
        seed: Random seed for reproducibility
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    images_dir = source / "images"
    labels_dir = source / "labels"
    source_yaml = source / "dataset.yaml"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Dataset not found at {source_dir}")
        print(f"   Expected: {images_dir} and {labels_dir}")
        return False
    
    if not source_yaml.exists():
        print(f"⚠️  dataset.yaml not found at {source_yaml}")
        print("   Will create a new one")
    
    # Create output structure
    train_images = output / "train" / "images"
    train_labels = output / "train" / "labels"
    val_images = output / "val" / "images"
    val_labels = output / "val" / "labels"
    
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return False
    
    print(f"📦 Found {len(image_files)} images in YOLO format")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    image_files_shuffled = image_files.copy()
    random.shuffle(image_files_shuffled)
    
    # Split
    split_idx = int(len(image_files_shuffled) * train_split)
    train_images_list = image_files_shuffled[:split_idx]
    val_images_list = image_files_shuffled[split_idx:]
    
    print(f"   Train: {len(train_images_list)} images ({train_split*100:.0f}%)")
    print(f"   Val: {len(val_images_list)} images ({(1-train_split)*100:.0f}%)")
    
    # Copy images and labels
    print("\n📋 Organizing dataset...")
    
    copied_train = 0
    copied_val = 0
    
    for img_file in train_images_list:
        # Copy image
        shutil.copy2(img_file, train_images / img_file.name)
        
        # Copy corresponding label
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, train_labels / label_file.name)
            copied_train += 1
        else:
            print(f"⚠️  Label not found for {img_file.name}")
    
    for img_file in val_images_list:
        # Copy image
        shutil.copy2(img_file, val_images / img_file.name)
        
        # Copy corresponding label
        label_file = labels_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, val_labels / label_file.name)
            copied_val += 1
        else:
            print(f"⚠️  Label not found for {img_file.name}")
    
    # Create/update dataset.yaml for YOLO training
    if source_yaml.exists():
        with open(source_yaml, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Create default config
        config = {
            'names': {
                0: 'bread_or_Roti_naan',
                1: 'curry_dish',
                2: 'rice_dish',
                3: 'dry_vegetable',
                4: 'snack_item',
                5: 'sweet_item',
                6: 'accompaniment',
                7: 'Dal_or_sambar',
                8: 'drink',
                9: 'eggs',
                10: 'fish_dish',
                11: 'fruits',
                12: 'pasta',
                13: 'salad',
                14: 'soup',
                15: 'south_indian_breakfast'
            }
        }
    
    # Update paths for training (absolute paths)
    config['path'] = str(output.absolute())
    config['train'] = 'train/images'
    config['val'] = 'val/images'
    config['nc'] = len(config.get('names', {}))
    
    # Save updated yaml
    output_yaml = output / "dataset.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created {output_yaml}")
    print(f"✅ Dataset organized at {output_dir}")
    print(f"   Train: {copied_train} images with labels")
    print(f"   Val: {copied_val} images with labels")
    print(f"\n📝 Ready to train YOLO with:")
    print(f"   python scripts/train_yolo_model.py --data {output_yaml}")
    
    return True


def verify_yolo_dataset(dataset_dir="data/yolo_training_data"):
    """Verify YOLO dataset structure"""
    dataset_path = Path(dataset_dir)
    
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"
    dataset_yaml = dataset_path / "dataset.yaml"
    
    if not all([train_images.exists(), train_labels.exists(), val_images.exists(), val_labels.exists()]):
        print(f"❌ Dataset structure incomplete at {dataset_dir}")
        return False
    
    train_img_count = len(list(train_images.glob("*.jpg")) + list(train_images.glob("*.png")))
    train_label_count = len(list(train_labels.glob("*.txt")))
    val_img_count = len(list(val_images.glob("*.jpg")) + list(val_images.glob("*.png")))
    val_label_count = len(list(val_labels.glob("*.txt")))
    
    print("\n" + "=" * 60)
    print("YOLO Dataset Verification")
    print("=" * 60)
    print(f"Train Images: {train_img_count}")
    print(f"Train Labels: {train_label_count}")
    print(f"Val Images: {val_img_count}")
    print(f"Val Labels: {val_label_count}")
    
    if train_img_count == train_label_count and val_img_count == val_label_count:
        print("\n✅ Dataset structure is correct!")
        if dataset_yaml.exists():
            print(f"✅ dataset.yaml found at {dataset_yaml}")
        return True
    else:
        print("\n⚠️  Warning: Image/Label count mismatch")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "prepare":
            prepare_yolo_dataset()
        elif command == "verify":
            verify_yolo_dataset()
        else:
            print("Usage:")
            print("  python prepare_yolo_dataset.py prepare  # Prepare dataset")
            print("  python prepare_yolo_dataset.py verify    # Verify dataset")
    else:
        print("Preparing YOLO dataset...")
        if prepare_yolo_dataset():
            print("\nVerifying dataset...")
            verify_yolo_dataset()
