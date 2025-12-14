# Dataset Download Guide - Food-101

## Important Note ⚠️

**Food-101 has limited Indian food categories!**

Food-101 only has:
- ✅ `samosa` - Direct match
- ✅ `chicken_curry` - Can use for "Curry"

**Missing from Food-101:**
- Biryani, Dosa, Idli, Naan, Roti, Dal, Paneer Tikka, Butter Chicken, Palak Paneer, Chole, Rajma, Aloo Gobi, Baingan Bharta

## Options

### Option 1: Use Food-101 + Supplement (Recommended)

1. **Download Food-101** (get samosa + curry)
2. **Find Indian Food Dataset** on Kaggle for other foods
3. **Combine datasets** using our scripts

### Option 2: Use Multiple Kaggle Datasets

Search Kaggle for:
- "Indian food dataset"
- "Indian cuisine images"
- "Food classification dataset"

### Option 3: Manual Collection

- Take photos
- Download from food blogs (with permission)
- Use stock photos

## Quick Start with Food-101

### Step 1: Install Kaggle API (Optional)

```bash
pip install kaggle
```

### Step 2: Set Up Kaggle Credentials

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`

### Step 3: Download Food-101

**Option A: Using Script (if Kaggle API set up)**
```bash
python scripts/download_food101.py download
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/dansbecker/food101
2. Click "Download" (requires Kaggle account)
3. Extract zip file

### Step 4: Extract Relevant Images

```bash
python scripts/download_food101.py extract /path/to/food-101
```

This will extract:
- `samosa` → `data/training_data/train/Samosa/`
- `chicken_curry` → `data/training_data/train/Curry/`

### Step 5: Supplement with Other Sources

For the remaining 13 food classes, you'll need other sources.

## Better Alternative: Indian Food Dataset

Since Food-101 has limited Indian foods, consider:

1. **Search Kaggle for "Indian food dataset"**
   - There are several Indian food-specific datasets
   - These will have more relevant categories

2. **Use Multiple Sources:**
   - Food-101 for samosa, curry
   - Indian Food Images dataset for others
   - Combine using our scripts

## Combining Multiple Datasets

Once you have images from multiple sources:

```bash
# Organize first dataset
python scripts/prepare_dataset.py organize /path/to/dataset1

# Organize second dataset (will add to existing)
python scripts/prepare_dataset.py organize /path/to/dataset2

# Verify combined dataset
python scripts/prepare_dataset.py verify
```

## Recommended Approach

1. **Start with Food-101** (get samosa + curry)
2. **Search Kaggle** for "Indian food images" or "Indian cuisine dataset"
3. **Download best matching dataset**
4. **Combine datasets** using our scripts
5. **Fill gaps** with manual collection if needed

## Next Steps

After getting images:
1. Organize: `python scripts/prepare_dataset.py organize <path>`
2. Verify: `python scripts/prepare_dataset.py verify`
3. Train: `python scripts/train_vision_model.py`

---

**Note:** Food-101 alone won't be enough. You'll need to supplement with other sources for a complete Indian food dataset.

