# Khana Dataset Setup Guide

This guide will help you download and set up the Khana dataset to **completely replace** the current dataset in your Food Vision AI project.

## Classification Format

This project uses **image classification** for dish recognition:
- ✅ Optimized for **single-dish recognition** (identifies one dish per image)
- ✅ Uses **EfficientNet/ResNet** classification models
- ✅ Expects **classification format** dataset structure
- ✅ Format: `train/class1/`, `train/class2/`, `val/class1/`, etc.

**The Khana dataset is in classification format** - perfect for this project!

## What Gets Replaced

When you set up the Khana dataset:
- ✅ **Old dataset will be completely removed** from `data/training_data/`
- ✅ **Old dataset will be backed up** (just in case)
- ✅ **New Khana dataset** will be placed in `data/training_data/`
- ✅ **App, UI, and nutrition features stay the same** - only the training data changes

## Prerequisites

1. **Google Drive File ID**: You need the file ID from the Google Drive link to the Khana dataset ZIP file.
   - The file ID is the part after `/d/` in the Google Drive URL
   - Example: If the URL is `https://drive.google.com/file/d/1ABC123xyz.../view`, the file ID is `1ABC123xyz...`

2. **Install gdown** (if not already installed):
   ```bash
   pip install gdown
   # Or if you have permission issues:
   python3 -m pip install --break-system-packages gdown
   ```

## Step 1: Download the Dataset

Run the download script with your Google Drive file ID:

```bash
cd /export/home/4prasad/piyush/Food_Vision_AI
./scripts/download_khana_dataset.sh <YOUR_FILE_ID>
```

Or with the full Google Drive URL:
```bash
./scripts/download_khana_dataset.sh "https://drive.google.com/file/d/<YOUR_FILE_ID>/view"
```

The script will:
- Download the ZIP file to `data/downloads/khana.zip`
- Extract it to `data/khana_dataset/`

## Step 2: Organize and Configure the Dataset

After downloading, run the setup script to organize the dataset into classification format:

```bash
python3 scripts/setup_khana_dataset.py
```

This script will:
1. Analyze the dataset structure
2. Extract class names from class folders
3. Organize the data into classification format (train/class1/, val/class1/, etc.)
4. Create train/val/test splits (80/10/10)
5. Save class names to `class_names.txt`
6. Backup your existing dataset (if any)

## Step 3: Verify the Setup

Check that everything is set up correctly:

```bash
# Check the dataset structure
ls -la data/training_data/

# Check class names
cat data/training_data/class_names.txt

# Count images per split
find data/training_data/train -type f | wc -l
find data/training_data/val -type f | wc -l
find data/training_data/test -type f | wc -l
```

## Manual Steps (if needed)

If the automatic setup doesn't work perfectly, you may need to:

1. **Inspect the dataset structure**:
   ```bash
   ls -la data/khana_dataset/
   ```

2. **Manually organize** if the dataset isn't in classification format:
   - Create `train/class1/`, `train/class2/`, etc.
   - Create `val/class1/`, `val/class2/`, etc.
   - Create `test/class1/`, `test/class2/`, etc.
   - Move/copy images to respective class folders

3. **Create class_names.txt manually** with one class name per line

## Troubleshooting

### gdown installation issues
If you can't install gdown system-wide, try:
- Using a virtual environment
- Using `pipx install gdown`
- Downloading manually from Google Drive and placing the ZIP in `data/downloads/khana.zip`

### Dataset structure issues
If the automatic organization fails:
1. Check the structure of `data/khana_dataset/`
2. The setup script will print what it found
3. Expected format: class folders with images inside, or train/val/test with class folders

### Class names not detected
If class names aren't automatically detected:
- Check if dataset has class folders (one folder per dish type)
- Or check if dataset is already split into train/val/test with class folders

## Step 4: Train Your Model

After setup is complete, train your classification model:

```bash
python scripts/train_classification_model.py \
    --data data/training_data \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32
```

This will:
- Train EfficientNet-B0 on Khana dataset
- Save model to `models/weights/food_classifier.pt`
- Save class names to `models/weights/class_names.txt`

The app will automatically use the trained model once it's saved!

## Summary

✅ **What gets set up:**
- Dataset organized in `data/training_data/` (train/val/test structure)
- Class names saved to `class_names.txt`
- Ready for model training

✅ **After training:**
- Model saved to `models/weights/food_classifier.pt`
- Class names saved to `models/weights/class_names.txt`
- App automatically uses the trained model

✅ **Features:**
- App UI and functionality
- Nutrition database (`data/nutrition_db.csv`)
- Classification model for dish recognition
- GenAI integration for descriptions

