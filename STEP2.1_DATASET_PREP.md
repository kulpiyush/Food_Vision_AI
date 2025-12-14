# Step 2.1: Prepare Fine-tuning Dataset

## Current Status
- ❌ No training dataset prepared
- ❌ `data/training_data/` directory is empty
- ❌ Need to collect and organize images

## What We Need

### Dataset Requirements:
- **Minimum:** 50-100 images per food category
- **Categories:** 15 Indian foods (matching our class list)
- **Total:** ~750-1500 images minimum
- **Split:** 80% train, 20% validation

### Food Categories (from INDIAN_FOOD_CLASSES):
1. Biryani
2. Dosa
3. Idli
4. Samosa
5. Curry
6. Naan
7. Roti
8. Dal
9. Paneer Tikka
10. Butter Chicken
11. Palak Paneer
12. Chole
13. Rajma
14. Aloo Gobi
15. Baingan Bharta

## Dataset Structure

```
data/training_data/
├── train/
│   ├── biryani/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── dosa/
│   ├── idli/
│   └── ... (all 15 categories)
└── val/
    ├── biryani/
    ├── dosa/
    └── ... (all 15 categories)
```

## Options for Getting Data

### Option 1: Download from Public Datasets (Recommended)
- **Food-101**: https://www.kaggle.com/datasets/dansbecker/food101
- **Indian Food Images**: Search Kaggle for "Indian food dataset"
- **Google Images**: Use image scraping tools (with permission)

### Option 2: Manual Collection
- Take photos of Indian food
- Download from food blogs/websites (with permission)
- Use stock photo sites

### Option 3: Synthetic Data (Advanced)
- Use data augmentation to expand small dataset
- Generate variations of existing images

## Next Steps

1. **Create dataset structure script** (we'll do this)
2. **Download/collect images** (you'll need to do this)
3. **Organize images into folders** (we'll create a script)
4. **Split into train/val** (we'll create a script)
5. **Verify dataset** (we'll create a script)

---

**Let's start by creating the dataset preparation scripts!**

