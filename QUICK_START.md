# Quick Start Guide - Get Your Project Running

Follow these steps to get FoodVisionAI up and running with the Khana dataset.

## Step 1: Install Dependencies

```bash
cd /export/home/4prasad/piyush/Food_Vision_AI
pip install -r requirements.txt
```

**Note**: If you encounter permission issues, use:
```bash
pip install --user -r requirements.txt
# OR
python3 -m pip install --break-system-packages -r requirements.txt
```

## Step 2: Get Khana Dataset

You need the **Google Drive file ID** for the Khana dataset ZIP file.

### How to get the file ID:
1. Open the Google Drive link to the Khana dataset
2. Look at the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
3. Copy the `FILE_ID_HERE` part (the long string after `/d/`)

### Download the dataset:
```bash
./scripts/download_khana_dataset.sh <YOUR_FILE_ID>
```

**Example:**
```bash
./scripts/download_khana_dataset.sh 1ABC123xyz456DEF789
```

Or with the full URL:
```bash
./scripts/download_khana_dataset.sh "https://drive.google.com/file/d/1ABC123xyz456DEF789/view"
```

**What happens:**
- Downloads the ZIP file to `data/downloads/khana.zip`
- Extracts it to `data/khana_dataset/`
- Backs up any existing dataset

## Step 3: Organize the Dataset

After downloading, organize the dataset into train/val/test structure:

```bash
python3 scripts/setup_khana_dataset.py
```

**What happens:**
- Analyzes the Khana dataset structure
- Organizes into `data/training_data/train/`, `val/`, `test/`
- Creates `class_names.txt` with all dish classes
- Shows dataset statistics

**Expected output:**
```
✅ Setup Complete!
Dataset location: data/training_data
Classes: 80
Train images: ~104,000
Val images: ~13,000
Test images: ~13,000
```

## Step 4: Train the Model

Train your classification model on the Khana dataset:

```bash
python scripts/train_classification_model.py \
    --data data/training_data \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

**Training options:**
- `--model`: Choose `efficientnet_b0` (recommended), `resnet50`, or `mobilenet_v2`
- `--epochs`: Number of training epochs (50 is a good start)
- `--batch-size`: Batch size (adjust based on your GPU memory)
- `--lr`: Learning rate (0.001 is default)

**What happens:**
- Trains the model on your dataset
- Saves model to `models/weights/food_classifier.pt`
- Saves class names to `models/weights/class_names.txt`
- Creates training history JSON

**Training time:**
- CPU: ~2-4 hours for 50 epochs
- GPU: ~30-60 minutes for 50 epochs

## Step 5: Run the Application

Once training is complete, start the Streamlit app:

```bash
streamlit run app.py
```

The app will:
- Automatically load your trained model
- Open in your browser (usually at `http://localhost:8501`)
- Be ready to classify Indian food images!

## Step 6: Test the App

1. **Upload an image** of Indian food (Biryani, Dosa, etc.)
2. **Click "Analyze Food"**
3. **View results:**
   - Detected dish name
   - Confidence score
   - Nutritional information
   - AI description (if Ollama is set up)

## Optional: Set Up GenAI (for AI Descriptions)

If you want AI-powered descriptions and Q&A:

1. **Install Ollama:**
   ```bash
   # Visit https://ollama.ai and follow installation instructions
   ```

2. **Download LLM model:**
   ```bash
   ollama pull llama3.2
   ```

3. **Restart the app** - GenAI features will be automatically enabled

See `OLLAMA_SETUP.md` for detailed instructions.

## Troubleshooting

### Issue: "gdown not found"
```bash
pip install gdown
# OR
python3 -m pip install --break-system-packages gdown
```

### Issue: "Permission denied" on download script
```bash
chmod +x scripts/download_khana_dataset.sh
```

### Issue: Dataset structure not recognized
- Check that the Khana dataset has class folders (one folder per dish type)
- Or check if it's already split into train/val/test
- Run `ls -la data/khana_dataset/` to inspect structure

### Issue: Out of memory during training
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use a smaller model: `--model mobilenet_v2`

### Issue: Model not loading in app
- Check that `models/weights/food_classifier.pt` exists
- Verify `models/weights/class_names.txt` exists
- Check file permissions

## Next Steps After Setup

1. **Improve accuracy**: Train for more epochs or use a larger model
2. **Add more dishes**: Expand the nutrition database
3. **Customize UI**: Modify `app.py` for your needs
4. **Deploy**: Deploy to Streamlit Cloud or your own server

## Quick Reference

```bash
# Download dataset
./scripts/download_khana_dataset.sh <FILE_ID>

# Setup dataset
python3 scripts/setup_khana_dataset.py

# Train model
python scripts/train_classification_model.py --data data/training_data --model efficientnet_b0 --epochs 50

# Run app
streamlit run app.py
```

---

**Ready to start? Begin with Step 1!** 🚀

