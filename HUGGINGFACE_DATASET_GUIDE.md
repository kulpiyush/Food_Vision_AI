# HuggingFace Dataset Guide - SOHL Multi-Dish Indian Food Dataset

## ✅ Great Find!

This dataset is perfect for our project! It has:
- ✅ 377 images of Indian food
- ✅ 16 food categories (some match our needs)
- ✅ Multi-dish detection (can extract individual foods)
- ✅ Real-world restaurant/home environments

## ⚠️ Important Notes

### Format Conversion Needed:
- **Dataset format**: YOLO (object detection with bounding boxes)
- **Our need**: Classification (ImageFolder style)
- **Solution**: We'll convert it! ✅

### Class Mapping:
The dataset has broader categories. We'll map them:

| Dataset Class | Maps to Our Classes |
|---------------|-------------------|
| `bread_or_Roti_naan` | Naan, Roti |
| `curry_dish` | Curry, Butter Chicken, Palak Paneer |
| `rice_dish` | Biryani |
| `snack_item` | Samosa |
| `Dal_or_sambar` | Dal |
| `south_indian_breakfast` | Dosa, Idli |
| `dry_vegetable` | Aloo Gobi, Baingan Bharta |

**Missing from dataset:**
- Paneer Tikka, Chole, Rajma (may need to supplement)

### Dataset Size:
- 377 images total
- With multi-dish detection, we can extract more crops
- Expected: ~500-1000 cropped images after conversion
- Average: ~30-60 images per class (may need more)

## Quick Start

### Step 1: Install Dependencies

```bash
pip install huggingface_hub pyyaml
```

### Step 2: Download and Convert (One Command!)

```bash
python scripts/download_huggingface_dataset.py full
```

This will:
1. Download dataset from HuggingFace
2. Convert YOLO format to classification format
3. Extract individual food items from images
4. Organize into train/val folders
5. Map to our food classes

### Step 3: Verify Dataset

```bash
python scripts/prepare_dataset.py verify
```

### Step 4: Supplement if Needed

If some classes have too few images:
- Search for additional images
- Use data augmentation (already in training script)
- Or proceed with what you have

### Step 5: Train Model

```bash
python scripts/train_vision_model.py
```

## Manual Steps (If Needed)

### Download Only:
```bash
python scripts/download_huggingface_dataset.py download
```

### Convert Only:
```bash
python scripts/download_huggingface_dataset.py convert data/hf_dataset
```

## What the Script Does

1. **Downloads** dataset from HuggingFace
2. **Reads** YOLO annotations (bounding boxes)
3. **Crops** individual food items from images
4. **Maps** dataset classes to our classes
5. **Splits** into train/val (80/20)
6. **Saves** in classification format

## Expected Results

After conversion:
- ✅ Multiple food crops per image (since it's multi-dish)
- ✅ Organized by our class names
- ✅ Train/val split ready
- ⚠️ Some classes may have fewer images (can supplement)

## Supplementing the Dataset

If you need more images for specific classes:

1. **Use Food-101** for Samosa + Curry
2. **Search Kaggle** for "Indian food [class name]"
3. **Manual collection** for rare classes
4. **Data augmentation** (training script includes this)

## Advantages of This Dataset

✅ **Real-world images**: Restaurant and home environments  
✅ **Multi-dish**: Can extract multiple foods per image  
✅ **Indian cuisine focus**: Perfect for our project  
✅ **Good quality**: Well-annotated bounding boxes  
✅ **Ready to use**: Just download and convert!

## Next Steps

1. **Download and convert** (one command!)
2. **Verify** dataset
3. **Supplement** if needed (optional)
4. **Train** model
5. **Test** improved accuracy!

---

**This dataset is a great starting point! Let's download and convert it!** 🚀

