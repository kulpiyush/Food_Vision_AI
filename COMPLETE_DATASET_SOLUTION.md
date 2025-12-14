# Complete Dataset Solution - Food-101 + Supplements

## ⚠️ Important: Food-101 Limitation

**Food-101 only has 2 out of 15 Indian food categories we need:**
- ✅ `samosa` - Direct match
- ✅ `chicken_curry` - Can use for "Curry"

**Missing 13 categories:**
- Biryani, Dosa, Idli, Naan, Roti, Dal, Paneer Tikka, Butter Chicken, Palak Paneer, Chole, Rajma, Aloo Gobi, Baingan Bharta

## Recommended Solution: Multi-Source Approach

### Step 1: Download Food-101 (Get 2 categories)

**Option A: Using Kaggle API**
```bash
# Install Kaggle
pip install kaggle

# Set up credentials (get from https://www.kaggle.com/account)
# Save to ~/.kaggle/kaggle.json

# Download
python scripts/download_food101.py download
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/dansbecker/food101
2. Click "Download" (requires Kaggle account)
3. Extract zip file

**Extract relevant images:**
```bash
python scripts/download_food101.py extract /path/to/food-101
```

This gives you: **Samosa** and **Curry** ✅

### Step 2: Find Indian Food Dataset (Get remaining 13)

**Search Kaggle for:**
- "Indian food dataset"
- "Indian cuisine images"
- "Indian food classification"

**Recommended datasets to check:**
1. Search: "Indian food images" on Kaggle
2. Look for datasets with multiple Indian food categories
3. Download the one with most matching categories

### Step 3: Combine Datasets

Once you have images from multiple sources:

```bash
# Organize Food-101 images (already done if you ran extract)
# Organize Indian food dataset
python scripts/prepare_dataset.py organize /path/to/indian_food_dataset

# Verify combined dataset
python scripts/prepare_dataset.py verify
```

### Step 4: Fill Gaps (If needed)

For any remaining missing categories:

**Option A: Manual Collection**
- Take photos of food
- Download from food blogs (with permission)
- Use stock photos

**Option B: Use Similar Categories**
- Use "fried_rice" from Food-101 as starting point for Biryani
- Use "bread" categories for Naan/Roti

**Option C: Data Augmentation**
- If you have some images, use augmentation to expand
- Our training script includes augmentation

## Quick Start Guide

### 1. Download Food-101
```bash
# If you have Kaggle API set up:
python scripts/download_food101.py download

# Or download manually from Kaggle website
```

### 2. Extract from Food-101
```bash
python scripts/download_food101.py extract /path/to/food-101
```

### 3. Search for Indian Food Dataset
- Go to https://www.kaggle.com/datasets
- Search: "Indian food"
- Download best matching dataset

### 4. Organize All Images
```bash
# Organize Indian food dataset
python scripts/prepare_dataset.py organize /path/to/indian_food_dataset

# Verify
python scripts/prepare_dataset.py verify
```

### 5. Train Model
```bash
python scripts/train_vision_model.py
```

## Alternative: Start with Smaller Dataset

If you can't find a complete dataset, you can:

1. **Start with what you have** (even if incomplete)
2. **Train on available categories** (e.g., just Samosa + Curry)
3. **Add more categories later** as you find them
4. **Re-train** with expanded dataset

The model will work, just with fewer food categories initially.

## Minimum Requirements

For each food class, you need:
- **Minimum:** 30-50 images (can work with less)
- **Recommended:** 50-100 images
- **Ideal:** 100+ images

**Total minimum:** ~450-750 images (30-50 per class × 15 classes)

## What We Have Ready

✅ **Scripts created:**
- `scripts/download_food101.py` - Download/extract Food-101
- `scripts/prepare_dataset.py` - Organize any dataset
- `scripts/train_vision_model.py` - Train model

✅ **Structure ready:**
- `data/training_data/train/` - Ready for images
- `data/training_data/val/` - Ready for images

## Next Steps

1. **Download Food-101** (get Samosa + Curry)
2. **Find Indian food dataset** on Kaggle
3. **Combine using our scripts**
4. **Train model**

---

**The scripts are ready! Just need to download the datasets.** 🚀

