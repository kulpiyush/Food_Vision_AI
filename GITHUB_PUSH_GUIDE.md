# GitHub Push Guide

## ✅ What Will Be Pushed

**Small files (included):**
- ✅ All source code (`.py`, `.sh`, `.md` files)
- ✅ `data/nutrition_db.csv` (600 bytes) - Nutrition database
- ✅ `models/weights/food_classifier.pt` (16MB) - Trained model
- ✅ `models/weights/class_names.txt` (854 bytes) - Class names
- ✅ `requirements.txt` - Dependencies
- ✅ All documentation files

**Large files (excluded via .gitignore):**
- ❌ `data/training_data/` (~6.8GB) - Training dataset
- ❌ `data/khana_dataset/` (~6.8GB) - Extracted dataset
- ❌ `data/downloads/` (~6.5GB) - Downloaded files
- ❌ `*.tar.gz`, `*.zip` - Archive files

## Repository Size Estimate

- **Code + docs**: ~1-2MB
- **Trained model**: 16MB
- **Nutrition DB**: <1KB
- **Total**: ~18-20MB (well under GitHub limits ✅)

## Steps to Push to GitHub

### 1. Initialize Git Repository (if not already done)

```bash
cd /export/home/4prasad/piyush/Food_Vision_AI
git init
```

### 2. Check What Will Be Committed

```bash
# See what files will be tracked
git status

# Verify large files are ignored
git status --ignored | grep -E "training_data|khana_dataset|downloads"
```

### 3. Add Files

```bash
# Add all files (large ones will be automatically ignored)
git add .

# Verify what's staged
git status
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: FoodVisionAI - Indian food classification app

- Classification model with 93.25% accuracy
- 80 Indian dish classes
- Nutrition database integration
- GenAI support for descriptions
- Trained on Khana dataset (105K+ images)"
```

### 5. Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `FoodVisionAI`)
3. **DO NOT** initialize with README (you already have one)

### 6. Connect and Push

```bash
# Add remote (replace with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/FoodVisionAI.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Verify After Push

1. Check repository size on GitHub (should be ~20MB, not 20GB)
2. Verify `data/training_data/` is NOT in the repository
3. Verify `models/weights/food_classifier.pt` IS included
4. Verify `data/nutrition_db.csv` IS included

## Important Notes

### ⚠️ Dataset Not Included

The training dataset is **NOT** included in the repository. Users need to:
1. Download from Google Drive (see `DATASET_SETUP.md`)
2. Run setup scripts to organize the dataset
3. The app works with the pre-trained model immediately

### ✅ Pre-trained Model Included

The trained model (`food_classifier.pt`) **IS** included because:
- It's only 16MB (acceptable for GitHub)
- Users can use the app immediately without training
- Essential for the app to work

### 📝 Documentation

Make sure to include:
- `README.md` - Main project documentation
- `DATASET_SETUP.md` - Instructions to download dataset
- `KHANA_DATASET_SETUP.md` - Detailed setup guide
- `QUICK_START.md` - Quick start instructions

## Troubleshooting

### If you see large files being tracked:

```bash
# Check .gitignore is working
git check-ignore -v data/training_data/

# If files are already tracked, remove them:
git rm -r --cached data/training_data/
git rm -r --cached data/khana_dataset/
git rm -r --cached data/downloads/
git commit -m "Remove large dataset files"
```

### If push fails due to size:

```bash
# Check repository size
du -sh .git

# If too large, consider using Git LFS for model weights:
# git lfs install
# git lfs track "*.pt"
# git add .gitattributes
```

## Summary

✅ **Safe to push**: Repository will be ~20MB (not 20GB)
✅ **Large datasets excluded**: Via .gitignore
✅ **Essential files included**: Model, nutrition DB, code
✅ **Users can download dataset**: Instructions provided

You're ready to push! 🚀

