# Step 2.2: Fine-tune Vision Model

## Training Script Created ✅

I've created `scripts/train_vision_model.py` - a complete training script ready to use once you have images.

## What the Script Does

1. **Loads Dataset**
   - Reads from `data/training_data/train/` and `data/training_data/val/`
   - Applies data augmentation (rotation, flip, color jitter)
   - Creates train/val data loaders

2. **Creates Model**
   - Loads pretrained EfficientNet-B0
   - Modifies classifier head for your number of classes
   - Sets up optimizer and learning rate scheduler

3. **Trains Model**
   - Transfer learning (uses pretrained backbone)
   - Fine-tunes on Indian cuisine
   - Saves best model based on validation accuracy

4. **Saves Model**
   - Saves to `models/weights/food_classifier_indian.pth`
   - Includes model weights, class names, and metadata

## How to Use

### Step 1: Prepare Dataset
```bash
# Make sure you have images in data/training_data/
python scripts/prepare_dataset.py verify
```

### Step 2: Train Model
```bash
# Basic training (15 epochs, batch size 32)
python scripts/train_vision_model.py

# Custom training
python scripts/train_vision_model.py \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0005 \
    --data_dir data/training_data \
    --output models/weights/food_classifier_indian.pth
```

### Step 3: Use Fine-tuned Model
The app will automatically use the fine-tuned model if it exists at:
`models/weights/food_classifier_indian.pth`

## Training Parameters

- **Epochs**: 15 (default), adjust based on dataset size
- **Batch Size**: 32 (default), reduce if out of memory
- **Learning Rate**: 0.001 (default), with step scheduler
- **Device**: Auto-detects CUDA/CPU

## Expected Results

After training, you should see:
- **Validation Accuracy**: 70-90%+ (depending on dataset quality)
- **Confidence Scores**: 70-95% (much higher than current 10%)
- **Better Predictions**: Model actually recognizes Indian foods

## Current Status

- ✅ Training script created
- ✅ Ready to use
- ❌ Waiting for dataset images

## Next Steps

1. **Collect Images** (Step 2.1)
   - Get 50-100 images per food class
   - Organize in train/val folders

2. **Run Training** (Step 2.2)
   ```bash
   python scripts/train_vision_model.py
   ```

3. **Test Improved Model**
   - Run app: `streamlit run app.py`
   - Upload food images
   - See much higher confidence scores!

---

**The training script is ready! Just need images to train on.** 🚀

