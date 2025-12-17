# FoodVisionAI - Indian Food Recognition & Nutrition Analysis

An AI-powered application that identifies Indian dishes from images and provides detailed nutritional information using deep learning classification models and semantic matching.

## 🎯 Features

- 🍽️ **Food Recognition**: Accurately classifies Indian dishes from images using state-of-the-art deep learning models (EfficientNet, ResNet, MobileNet)
- 📊 **Nutrition Analysis**: Provides comprehensive nutritional information using Kaggle Indian Food Nutrition dataset (1014 foods) with semantic matching
- 🤖 **AI Descriptions**: Generates intelligent descriptions of dishes using local LLM (Ollama)
- 💬 **Interactive Q&A**: Ask questions about dishes, nutrition, and health benefits
- 🚀 **Fast Inference**: Optimized for real-time predictions (~14ms per food search)
- 🎨 **Modern UI**: Beautiful Streamlit-based web interface
`  
## 🏗️ Architecture

### System Overview & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                    │
│  • Image Upload (JPG/PNG)                                       │
│  • Model Selection (EfficientNet/ResNet/MobileNet)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              IMAGE PREPROCESSING                                 │
│  • Resize: 224×224 pixels                                        │
│  • Normalize: ImageNet stats                                    │
│  • Tensor conversion                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              VISION MODEL (PyTorch)                              │
│  Model: EfficientNet-B0 / ResNet-50 / MobileNet-V2             │
│  • Pre-trained: ImageNet (transfer learning)                    │
│  • Fine-tuned: Khana Dataset (131K+ images, 80 classes)        │
│  • Output: Logits (80 classes)                                  │
│  • Softmax: Probability distribution                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         OUT-OF-DISTRIBUTION (OOD) DETECTION                      │
│  • Calculate Entropy (uncertainty measure)                      │
│  • Calculate Confidence Gap (top-1 vs top-2)                   │
│  • Check Confidence Threshold                                    │
│  • Decision: Indian dish? → Continue / Reject                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
              [OOD Detected]    [Valid Indian Dish]
                    │                 │
                    ▼                 ▼
            ┌──────────────┐  ┌──────────────────────────────┐
            │ Show Warning │  │  EXTRACT TOP PREDICTION       │
            │ "Not Indian" │  │  • Food Name (e.g., "Biryani")│
            └──────────────┘  │  • Confidence Score           │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         SEMANTIC MATCHING (Sentence Transformers)                │
│  Model: paraphrase-multilingual-MiniLM-L12-v2                  │
│  • Encode Food Name → Embedding Vector (384-dim)               │
│  • Load Cached Embeddings (1,014 foods)                        │
│  • Cosine Similarity Search                                     │
│  • Find Best Match in Kaggle Dataset                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         NUTRITION DATABASE (Kaggle)                             │
│  Dataset: Indian_Food_Nutrition_Processed.csv                  │
│  • 1,014 Indian foods                                           │
│  • Columns: Calories, Protein, Carbs, Fats, Fiber              │
│  • Match: Semantic similarity (not keyword)                     │
│  • Output: Nutrition per 100g                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         GENERATIVE AI (Ollama - Local LLM)                      │
│  Model: Llama 3.2 (or configurable)                            │
│  • Generate Food Description                                    │
│  • Answer User Questions                                        │
│  • Health Benefits Analysis                                     │
│  • Recipe Suggestions                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTS DISPLAY                              │
│  • Detected Food Name                                           │
│  • Confidence Score                                             │
│  • Nutritional Information (Calories, Protein, Carbs, Fats)    │
│  • AI-Generated Description                                     │
│  • Interactive Q&A Interface                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technical Components

#### 1. Vision Pipeline (`models/vision_model.py`)
- **Input**: Raw image (any size)
- **Processing**: Resize → Normalize → Tensor
- **Model**: PyTorch CNN (EfficientNet-B0, ResNet-50, MobileNet-V2)
- **Training**: Pre-trained on ImageNet, fine-tuned on Khana dataset (131,000+ images, 80 classes)
- **Output**: 80-class probability distribution
- **Task**: Single-dish classification per image

#### 2. Out-of-Distribution (OOD) Detection System
- **Entropy Calculation**: Measures prediction uncertainty
- **Confidence Gap Analysis**: Compares top-1 vs top-2 predictions
- **Confidence Threshold**: Filters low-confidence predictions
- **Purpose**: Prevents false positives for non-Indian foods
- **Result**: Shows warning message instead of incorrect classification

#### 3. Semantic Matching Engine (`utils/nutrition_calculator.py`)
- **Model**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Process**: Encode food name → Embedding vector (384-dim) → Cosine similarity search
- **Data Source**: Kaggle Indian Food Nutrition dataset (1,014 foods)
- **Features**:
  - Semantic matching (handles variations, not just keywords)
  - Cached embeddings for fast lookup (~14ms search time)
  - 100% match rate for all 80 Khana classes
  - Similarity score for match confidence

#### 4. Nutrition Database
- **Dataset**: `Indian_Food_Nutrition_Processed.csv` (Kaggle)
- **Coverage**: 1,014 Indian foods
- **Columns**: Calories, Protein, Carbs, Fats, Fiber, Micronutrients
- **Matching**: Semantic similarity (handles name variations)
- **Output**: Nutrition values per 100g serving

#### 5. Generative AI Integration (`models/genai_model.py`)
- **Provider**: Ollama (local LLM, no API costs)
- **Model**: Llama 3.2 (configurable)
- **Features**:
  - Intelligent food descriptions
  - Interactive Q&A interface
  - Health benefit analysis
  - Recipe suggestions
- **Privacy**: All processing happens locally

#### 6. Web Interface (`app.py`)
- **Framework**: Streamlit
- **Features**:
  - Image upload
  - Real-time analysis
  - Nutrition display
  - AI chat interface

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Programming language
- **PyTorch 2.0+**: Deep learning framework
- **Streamlit 1.28+**: Web application framework
- **Pillow 10.0+**: Image processing
- **Pandas 2.0+**: Data manipulation

### Deep Learning
- **torchvision**: Pre-trained models (EfficientNet, ResNet, MobileNet)
- **timm**: Additional vision models
- **sentence-transformers**: Semantic embeddings for food matching
- **scikit-learn**: Cosine similarity calculations

### Data & APIs
- **Kaggle API**: Dataset download
- **Ollama**: Local LLM for descriptions

### Utilities
- **NumPy**: Numerical computations
- **OpenCV**: Image processing
- **Albumentations**: Data augmentation

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

### Step 1: Clone Repository
   ```bash
   git clone <repository-url>
cd Automated_Nutritional_Analysis_App
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Download Kaggle Nutrition Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/batthulavinay/indian-food-nutrition)
2. Place `Indian_Food_Nutrition_Processed.csv` in `data/` folder
   
### Step 5: Download Khana Dataset (for Training)
1. Get Google Drive file ID for Khana dataset
2. Run download script:
   ```bash
   ./scripts/download_khana_dataset.sh <GOOGLE_DRIVE_FILE_ID>
   ```

3. Organize dataset:
   ```bash
   python scripts/setup_khana_dataset.py
   ```

## 🎓 Model Training

### Training the Classification Model

Train your model on the Khana dataset:

```bash
python scripts/train_classification_model.py \
    --data data/khana_dataset \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output models/weights
```

### Training Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--model` | `efficientnet_b0`, `resnet50`, `mobilenet_v2` | Model architecture |
| `--epochs` | Integer (default: 50) | Number of training epochs |
| `--batch-size` | Integer (default: 32) | Batch size (adjust for GPU memory) |
| `--lr` | Float (default: 0.001) | Learning rate |
| `--data` | Path | Path to training data |
| `--output` | Path | Output directory for model weights |

### Model Selection Guide

- **EfficientNet-B0** (Recommended): Best balance of speed and accuracy
- **ResNet-50**: Higher accuracy, slower inference
- **MobileNet-V2**: Fastest, optimized for mobile devices

### Training Output

After training, you'll get:
- `food_classifier.pt` - Trained model weights
- `class_names.txt` - List of all 80 dish classes
- `training_history.json` - Training metrics and history

### Expected Training Time
- **CPU**: ~2-4 hours for 50 epochs
- **GPU**: ~30-60 minutes for 50 epochs

## 📊 Dataset

### Khana Dataset (Training)
- **Total Images**: 131,000+
- **Dish Classes**: 80 categories
- **Format**: Classification (ImageFolder structure)
- **Split**: Train (80%), Validation (10%), Test (10%)

### Dataset Structure
```
data/khana_dataset/
├── train/
│   ├── aloo_gobi/
│   ├── biryani/
│   ├── dosa/
│   └── ... (80 classes)
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Kaggle Nutrition Dataset (Runtime)
- **Total Foods**: 1,014 Indian dishes
- **Source**: Anuvaad Indian Nutrient Database (INDB)
- **Columns**: Dish Name, Calories, Protein, Carbs, Fats, Fiber, Micronutrients
- **Matching**: Semantic similarity (sentence transformers)

## 🚀 Usage

### Running the Application

1. **Start Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open browser**: App will open at `http://localhost:8501`

3. **Upload image**: Select an image of Indian food

4. **Analyze**: Click "Analyze Food" to get:
   - Detected dish name
   - Confidence score
   - Nutritional information (calories, protein, carbs, fat, fiber)
   - AI-generated description (if Ollama is configured)
   - Q&A interface

### Model Selection in App

Choose different models in the sidebar:
- **EfficientNet-B0** (Recommended) - Best balance
- **ResNet-50** - Higher accuracy
- **MobileNet-V2** - Fastest inference

## 📁 Project Structure

```
Automated_Nutritional_Analysis_App/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config/
│   └── config.yaml                 # Configuration file
├── models/
│   ├── vision_model.py             # Classification model wrapper
│   ├── genai_model.py              # GenAI integration (Ollama)
│   └── weights/                    # Trained model weights
│       ├── food_classifier.pt      # Trained model
│       └── class_names.txt         # Class labels (80 dishes)
├── data/
│   ├── Indian_Food_Nutrition_Processed.csv  # Kaggle nutrition dataset
│   └── khana_dataset/              # Training dataset
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── download_khana_dataset.sh   # Download Khana dataset
│   ├── setup_khana_dataset.py      # Organize dataset structure
│   └── train_classification_model.py # Training script
└── utils/
    └── nutrition_calculator.py     # Nutrition lookup with semantic matching
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Vision Model Settings
vision_model:
  name: "efficientnet_b0"
  model_path: "models/weights/food_classifier.pt"
  confidence_threshold: 0.5

# Generative AI Settings
genai:
  provider: "ollama"
  model_name: "llama3.2"
  base_url: "http://localhost:11434"

# Nutritional Database
nutrition_db:
  path: "data/Indian_Food_Nutrition_Processed.csv"
  similarity_threshold: 0.5
```

## 🤖 GenAI Setup (Optional)

For AI-powered descriptions and Q&A:

1. **Install Ollama**:
   - Visit https://ollama.ai
   - Follow installation instructions for your OS

2. **Download LLM Model**:
   ```bash
   ollama pull llama3.2
   ```

3. **Start Ollama** (if not running automatically):
   ```bash
   ollama serve
   ```

4. **Restart App**: GenAI features will be automatically enabled

## 📈 Performance Metrics

### Model Performance
- **Inference Speed**: ~50-100ms per image (CPU), ~10-20ms (GPU)
- **Accuracy**: 85-95%+ on validation set (varies by model)
- **Model Size**: ~20-50MB depending on architecture

### Nutrition Matching
- **Search Time**: ~14ms per food (with caching)
- **Match Rate**: 100% (80/80 foods found)
- **Average Similarity**: 0.86 (excellent semantic matching)
- **Dataset Coverage**: All 1,014 foods supported

### Caching
- **Model Loading**: ~3.4s (one-time, cached in memory)
- **Embedding Encoding**: ~5.1s (one-time, cached to disk)
- **Subsequent Searches**: ~14ms (very fast!)

## 🔧 Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure `models/weights/food_classifier.pt` exists
   - Train the model first using training script

2. **Nutrition data not found**:
   - Check `data/Indian_Food_Nutrition_Processed.csv` exists
   - Verify file path in config

3. **Ollama not working**:
   - Ensure Ollama is installed and running
   - Check `ollama serve` is running
   - Verify model is downloaded: `ollama list`

4. **Slow performance**:
   - Use GPU for faster inference
   - Reduce batch size if out of memory
   - Check embeddings cache exists

## 🧪 Testing

Test the nutrition calculator:
```bash
python -c "from utils.nutrition_calculator import get_nutrition; print(get_nutrition('idli'))"
```

Test the vision model:
```bash
python verify_model.py
```

## 📝 Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- streamlit>=1.28.0
- torch>=2.0.0
- torchvision>=0.15.0
- sentence-transformers>=2.2.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- pillow>=10.0.0

## 🚀 Future Enhancements

### Portion Size Estimation (Planned)

We're planning to add intelligent portion size estimation from images to provide more accurate calorie counts:

**Approach:**
- **Reference Object Detection**: Detect coins, credit cards, or utensils to estimate scale
- **Plate/Bowl Size Detection**: Automatically detect and classify plate/bowl sizes (small: 6", medium: 8", large: 10")
- **Food Segmentation**: Segment food from background to calculate area/volume
- **Volume-to-Weight Conversion**: Use food density database to convert estimated volume to weight
- **ML-Based Estimation**: Train models to directly estimate portion sizes from images

**Expected Accuracy:**
- Reference object method: ±15-20%
- Plate detection method: ±25-30%
- ML-based method: ±10-15% (with training data)

**Benefits:**
- More accurate calorie counting
- Personalized nutrition tracking
- Better portion awareness

**Status**: Planned for future release

### Other Planned Features
- Multi-food detection in single image
- Meal planning and recommendations
- Calorie tracking over time
- Integration with fitness apps
- Voice commands for hands-free operation

---

**Technologies**: PyTorch | Streamlit | Sentence Transformers | Ollama