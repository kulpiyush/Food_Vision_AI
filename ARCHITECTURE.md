# FoodVisionAI - System Architecture

## 1. Overview

FoodVisionAI is an AI-powered application that analyzes food images to provide nutritional information, meal descriptions, and dietary recommendations.

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Streamlit/Gradio Web App)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Image Upload │  │  Results     │  │  Q&A Chat Interface  │ │
│  │   & Display  │  │  Display     │  │                      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
└─────────┼──────────────────┼──────────────────────┼────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Main Application Controller                  │  │
│  │  - Image preprocessing                                     │  │
│  │  - Pipeline orchestration                                 │  │
│  │  - Error handling                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE AI MODULES                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. VISION MODEL (Food Detection & Classification)       │  │
│  │     - Pretrained: EfficientNet-B0/B2 or ResNet-50        │  │
│  │     - Fine-tuned on local cuisine dataset                │  │
│  │     - Output: Food items + confidence scores             │  │
│  │     - Optional: YOLO for multi-food detection            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. PORTION ESTIMATION (Optional Extended Feature)        │  │
│  │     - Depth estimation or reference object detection      │  │
│  │     - Volume/weight estimation                            │  │
│  │     - Output: Estimated portion size                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. GENERATIVE AI (Explanations & Recommendations)       │  │
│  │     - Open-source: Llama 3.2, Mistral, or Phi-3          │  │
│  │     - API fallback: OpenAI GPT-4o-mini or Anthropic      │  │
│  │     - Functions:                                          │  │
│  │       * Food description                                  │  │
│  │       * Nutritional analysis                             │  │
│  │       * Meal suggestions                                  │  │
│  │       * Healthy alternatives                              │  │
│  │       * Q&A about the meal                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Nutritional Database (CSV/JSON/SQLite)                  │  │
│  │  - Food items with nutritional values                    │  │
│  │  - Calories, Fat, Carbs, Protein, Fiber, etc.            │  │
│  │  - Per 100g or per serving                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Fine-tuning Dataset (Local Cuisine)                     │  │
│  │  - Images of local food items                            │  │
│  │  - Labeled with food categories                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Technology Stack

### Frontend/UI
- **Streamlit** (Recommended - easier, faster to build)
- Alternative: Gradio or Flask + React

### Vision Models
- **Primary**: EfficientNet-B0/B2 (lightweight, fast)
- **Alternative**: ResNet-50, Vision Transformer (ViT), CLIP
- **Multi-food**: YOLOv8 or YOLOv5 (for detection)

### Generative AI
- **Open-source**: 
  - Llama 3.2 (3B/7B) via Ollama or HuggingFace
  - Mistral 7B
  - Phi-3 (Microsoft)
- **API** (for final polish):
  - OpenAI GPT-4o-mini
  - Anthropic Claude 3 Haiku
  - Google Gemini Pro

### Data Storage
- **Nutritional Data**: CSV/JSON files or SQLite database
- **Model Weights**: Local storage or HuggingFace Hub

### Libraries
- PyTorch or TensorFlow/Keras
- Transformers (HuggingFace)
- PIL/Pillow (image processing)
- NumPy, Pandas
- Streamlit/Gradio

## 4. Data Flow

1. **Image Upload** → User uploads food image via UI
2. **Preprocessing** → Resize, normalize, augment if needed
3. **Food Detection** → Vision model predicts food items
4. **Portion Estimation** (Optional) → Estimate serving size
5. **Nutritional Lookup** → Query database for nutritional values
6. **Calculation** → Calculate total nutrition based on portion
7. **GenAI Processing** → Generate description, analysis, recommendations
8. **Display Results** → Show predictions, nutrition, and AI-generated content
9. **Q&A** → Handle user questions about the meal

## 5. Model Selection Rationale

### Vision Model: EfficientNet-B0
- **Why**: Balance between accuracy and speed
- **Size**: ~5MB, fast inference
- **Fine-tuning**: Easy with transfer learning
- **Alternative**: ResNet-50 if more accuracy needed

### Generative Model: Llama 3.2 3B (via Ollama)
- **Why**: Free, runs locally, good quality
- **Setup**: Easy with Ollama
- **Fallback**: GPT-4o-mini API for better quality

## 6. Extended Features Implementation

### Multi-food Detection
- Use YOLOv8 for object detection
- Detect multiple food items in one image
- Classify each detected item separately

### Portion Size Estimation
- Reference object method (e.g., plate size, fork)
- Depth estimation (if available)
- Volume estimation from bounding boxes
- User correction interface

### Correction Interface
- Allow users to select/remove detected items
- Manual portion size input
- Recalculate nutrition based on corrections

### Personalized Recommendations
- User profile (dietary goals, restrictions)
- Compare meal to daily targets
- Suggest modifications

## 7. Project Structure

```
Automated_Nutritional_Analysis_App/
├── app.py                          # Main Streamlit application
├── models/
│   ├── vision_model.py            # Vision model wrapper
│   ├── genai_model.py             # Generative AI wrapper
│   └── weights/                   # Saved model weights
├── data/
│   ├── nutrition_db.csv           # Nutritional database
│   └── training_data/              # Fine-tuning dataset
├── utils/
│   ├── image_processing.py        # Image preprocessing
│   ├── nutrition_calculator.py    # Nutrition calculations
│   └── portion_estimator.py       # Portion estimation (optional)
├── config/
│   └── config.yaml                # Configuration file
├── requirements.txt               # Dependencies
├── README.md                      # User documentation
├── ARCHITECTURE.md                # This file
└── TECHNICAL_REPORT.md            # Technical documentation
```

## 8. Implementation Phases

### Phase 1: Foundation (Days 1-3)
- Set up project structure
- Implement basic vision model (pretrained, no fine-tuning yet)
- Create nutritional database
- Basic UI with image upload

### Phase 2: Core Features (Days 4-7)
- Fine-tune vision model on local cuisine
- Integrate nutritional lookup
- Add generative AI for descriptions
- Improve UI

### Phase 3: Extended Features (Days 8-11)
- Multi-food detection
- Portion estimation
- Correction interface
- Personalized recommendations

### Phase 4: Polish & Documentation (Days 12-14)
- Optimize model performance
- Improve UI/UX
- Write documentation
- Prepare presentation

## 9. Performance Considerations

- **Model Optimization**: 
  - Quantization (INT8)
  - Model pruning
  - ONNX conversion for faster inference
  
- **Caching**: 
  - Cache model predictions
  - Cache GenAI responses for common queries

- **Async Processing**: 
  - Background processing for GenAI
  - Progress indicators

## 10. Limitations & Future Work

### Known Limitations
- Accuracy depends on training data quality
- Portion estimation may be approximate
- GenAI responses may need fact-checking

### Future Improvements
- Real-time video analysis
- Mobile app version
- Integration with fitness trackers
- Multi-language support
- Barcode scanning integration

