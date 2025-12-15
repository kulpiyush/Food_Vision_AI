# FoodVisionAI - Indian Food Recognition & Nutrition Analysis

An AI-powered application that identifies Indian dishes from images and provides detailed nutritional information using deep learning classification models.

## Features

- 🍽️ **Food Recognition**: Accurately classifies Indian dishes from images using state-of-the-art deep learning models
- 📊 **Nutrition Analysis**: Provides comprehensive nutritional information for detected dishes
- 🤖 **AI Descriptions**: Generates intelligent descriptions of dishes using local LLM (Ollama)
- 💬 **Interactive Q&A**: Ask questions about dishes, nutrition, and health benefits

## How It Works

1. **Upload** an image of Indian food
2. **Classify** the dish using a fine-tuned EfficientNet/ResNet model
3. **Retrieve** nutritional information from the database
4. **Generate** AI-powered descriptions and answer questions

## Architecture

### Classification Model
- **Base Models**: EfficientNet-B0, ResNet-50, or MobileNet-V2
- **Training Dataset**: Khana dataset (131,000+ images, 80 Indian dish classes)
- **Task**: Image classification (single dish per image)
- **Output**: Dish name with confidence score

### Technology Stack
- **Frontend**: Streamlit
- **Deep Learning**: PyTorch + torchvision
- **Model Architecture**: EfficientNet/ResNet (transfer learning from ImageNet)
- **GenAI**: Ollama (local LLM for descriptions)
- **Data Processing**: Pandas, PIL

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Food_Vision_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Khana Dataset**
   
   Get the Google Drive file ID for the Khana dataset, then:
   ```bash
   ./scripts/download_khana_dataset.sh <GOOGLE_DRIVE_FILE_ID>
   ```

4. **Organize Dataset**
   ```bash
   python3 scripts/setup_khana_dataset.py
   ```

   This will organize the dataset into the required structure:
   ```
   data/training_data/
   ├── train/
   │   ├── Biryani/
   │   ├── Dosa/
   │   ├── Idli/
   │   └── ... (80 classes)
   ├── val/
   │   └── ... (same classes)
   └── test/
       └── ... (same classes)
   ```

5. **Train the Model**
   ```bash
   python scripts/train_classification_model.py \
       --data data/training_data \
       --model efficientnet_b0 \
       --epochs 50 \
       --batch-size 32 \
       --lr 0.001
   ```

   The trained model will be saved to `models/weights/food_classifier.pt`

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Model Training

### Training Options

**Model Architectures:**
- `efficientnet_b0` - Recommended (best balance of speed and accuracy)
- `resnet50` - Higher accuracy, slower inference
- `mobilenet_v2` - Fastest, optimized for mobile devices

**Training Parameters:**
```bash
python scripts/train_classification_model.py \
    --data data/training_data \          # Dataset path
    --model efficientnet_b0 \            # Model architecture
    --epochs 50 \                        # Training epochs
    --batch-size 32 \                    # Batch size
    --lr 0.001 \                         # Learning rate
    --output models/weights              # Output directory
```

### Training Output

After training, you'll get:
- `food_classifier.pt` - Trained model weights
- `class_names.txt` - List of all dish classes
- `training_history.json` - Training metrics and history

## Dataset

### Khana Dataset

The project uses the **Khana dataset**, a comprehensive collection of Indian food images:
- **Total Images**: 131,000+
- **Dish Classes**: 80 categories
- **Format**: Classification (images organized by dish type)
- **Split**: Train (80%), Validation (10%), Test (10%)

### Dataset Structure

The dataset is organized in ImageFolder format:
```
data/training_data/
├── train/
│   ├── Biryani/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Dosa/
│   ├── Idli/
│   └── ... (80 classes)
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

## Usage

### Running the App

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload an image of Indian food through the web interface

3. Click "Analyze Food" to get:
   - Detected dish name
   - Confidence score
   - Nutritional information (calories, protein, carbs, fat)
   - AI-generated description (if Ollama is configured)
   - Q&A interface for additional questions

### Model Selection

You can choose different models in the app sidebar:
- **EfficientNet-B0** (Recommended) - Best balance
- **ResNet-50** - Higher accuracy
- **MobileNet-V2** - Fastest inference

## Project Structure

```
Food_Vision_AI/
├── app.py                              # Main Streamlit application
├── models/
│   ├── vision_model.py                # Classification model wrapper
│   └── weights/                        # Trained model weights
│       ├── food_classifier.pt         # Trained model
│       └── class_names.txt            # Class labels
├── data/
│   ├── training_data/                  # Khana dataset
│   │   ├── train/                     # Training images
│   │   ├── val/                       # Validation images
│   │   └── test/                      # Test images
│   └── nutrition_db.csv               # Nutrition database
├── scripts/
│   ├── download_khana_dataset.sh       # Download dataset
│   ├── setup_khana_dataset.py         # Organize dataset
│   └── train_classification_model.py   # Training script
└── utils/
    ├── image_processing.py            # Image utilities
    ├── nutrition_calculator.py        # Nutrition lookup
    └── genai_model.py                 # GenAI integration
```

## GenAI Integration (Optional)

For AI-powered descriptions and Q&A features:

1. **Install Ollama**
   - Visit https://ollama.ai
   - Follow installation instructions for your OS

2. **Download LLM Model**
   ```bash
   ollama pull llama3.2
   ```

3. **Restart the App**
   - The app will automatically detect Ollama
   - GenAI features will be enabled

See `OLLAMA_SETUP.md` for detailed setup instructions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- Streamlit 1.28+
- Pillow 10.0+
- pandas 2.0+
- numpy 1.24+

See `requirements.txt` for the complete list of dependencies.

## Performance

- **Inference Speed**: ~50-100ms per image (CPU), ~10-20ms (GPU)
- **Accuracy**: Varies by model (EfficientNet-B0 typically achieves 85-90%+ on validation set)
- **Model Size**: ~20-50MB depending on architecture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See individual files for licensing information.

---

**Built with ❤️ for Indian cuisine recognition and nutrition analysis**
