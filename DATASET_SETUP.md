# Dataset Setup Instructions

This repository does **NOT** include the training dataset due to size constraints (20GB+).

## Why Dataset is Not Included

- **Dataset size**: ~20GB (too large for GitHub)
- **GitHub limits**: 100MB per file, recommended repo size < 1GB
- **Best practice**: Large datasets should be downloaded separately

## How to Get the Dataset

### Step 1: Download Khana Dataset

The dataset is available on Google Drive:
- **Folder Link**: https://drive.google.com/drive/folders/1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0
- **Contains**: `dataset.tar.gz` (6.43 GB), `labels.txt`, `taxonomy.csv`

**Download using the provided script:**

```bash
./scripts/download_khana_from_folder.sh
```

Or manually:
```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0" -O data/downloads
```

### Step 2: Extract and Organize

```bash
# Extract dataset
cd data/downloads
tar -xzf dataset.tar.gz -C ../khana_dataset/
cd ../..

# Organize into train/val/test
python3 scripts/setup_khana_dataset.py
```

### Step 3: Verify Setup

```bash
# Check dataset structure
ls -la data/training_data/
# Should see: train/, val/, test/, class_names.txt
```

## What's Included in Repository

✅ **Small essential files** (included):
- `data/nutrition_db.csv` - Nutrition database (~600 bytes)
- `models/weights/class_names.txt` - Class names list
- `models/weights/food_classifier.pt` - Trained model (16MB)

❌ **Large files** (excluded via .gitignore):
- `data/training_data/` - Organized dataset (~20GB)
- `data/khana_dataset/` - Extracted dataset
- `data/downloads/` - Downloaded ZIP files

## Quick Start After Cloning

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Food_Vision_AI
   ```

2. **Download and setup dataset** (see steps above)

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

The trained model (`food_classifier.pt`) is included, so the app will work immediately after downloading the dataset.

## Alternative: Use Pre-trained Model Only

If you only want to use the app (not train), you can skip downloading the full dataset:
- The trained model is already included
- You just need `data/nutrition_db.csv` (included)
- The app will work with the pre-trained model

## Dataset Statistics

- **Total images**: ~148,000
- **Classes**: 80 Indian dishes
- **Train/Val/Test split**: 80/10/10
- **Model accuracy**: 93.25% validation accuracy

