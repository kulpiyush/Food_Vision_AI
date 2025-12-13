# FoodVisionAI - Automated Nutritional Analysis App

An AI-powered application that analyzes food images to provide nutritional information, meal descriptions, and dietary recommendations using deep learning and generative AI.

## 🎯 Project Overview

FoodVisionAI uses:
- **Vision Models** (EfficientNet/ResNet) for food detection and classification
- **Generative AI** (Llama/Mistral/GPT) for food descriptions and recommendations
- **Nutritional Database** for accurate nutritional information
- **Streamlit UI** for easy interaction

## ✨ Features

### Core Features
- ✅ Food image classification using fine-tuned vision models
- ✅ Automatic nutritional information retrieval
- ✅ AI-generated food descriptions and analysis
- ✅ Meal suggestions and healthy alternatives
- ✅ Interactive Q&A about meals

### Extended Features (Optional)
- 🔄 Multi-food detection in single image
- 🔄 Portion size estimation
- 🔄 Correction interface for inaccurate detections
- 🔄 Personalized dietary recommendations
- 🔄 Model optimization and quantization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- GPU recommended (optional, CPU works too)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Automated_Nutritional_Analysis_App
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama (for local GenAI):**
   ```bash
   # Visit https://ollama.ai and install Ollama
   # Then pull a model:
   ollama pull llama3.2
   ```

5. **Prepare nutritional database:**
   - Download or create `data/nutrition_db.csv`
   - Format: `food_name,calories,fat_g,carbs_g,protein_g,fiber_g,per_100g`

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Automated_Nutritional_Analysis_App/
├── app.py                    # Main Streamlit application
├── models/
│   ├── vision_model.py      # Vision model wrapper
│   ├── genai_model.py       # Generative AI wrapper
│   └── weights/             # Saved model weights
├── data/
│   ├── nutrition_db.csv    # Nutritional database
│   └── training_data/       # Fine-tuning dataset
├── utils/
│   ├── image_processing.py
│   ├── nutrition_calculator.py
│   └── portion_estimator.py
├── config/
│   └── config.yaml          # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── ARCHITECTURE.md         # System architecture
└── IMPLEMENTATION_GUIDE.md # Step-by-step guide
```

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed system architecture and design
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Step-by-step implementation guide

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Vision model selection
- Generative AI provider and model
- Nutritional database path
- Feature toggles

## 🎓 Assignment Details

**Module:** Data Analytics-3  
**Instructor:** Prof. Dr. Gayan de Silva  
**Deadline:** December 16th, 2025, 9am-1pm  
**Total Points:** 100

### Assessment Criteria
1. UI and Features (App design, usability, stability)
2. Extended Features (Creativity beyond requirements)
3. Model Efficiency & Improvements (Cost reduction, speed enhancements)
4. Presentation & PPT (Clarity, demonstration, explanation)
5. Documentation & Code Quality (Structure, README, comments, reproducibility)

## 🛠️ Development Roadmap

- [x] Architecture design
- [ ] Basic UI setup
- [ ] Vision model integration
- [ ] Nutritional database integration
- [ ] Generative AI integration
- [ ] Extended features
- [ ] Optimization and polish
- [ ] Documentation

## 📝 Notes

- Use open-source datasets and pretrained models
- Focus on intelligent system design and creative GenAI integration
- Fine-tune on local cuisine for better accuracy

## 🤝 Contributing

This is an individual project assignment. For questions or issues, refer to the implementation guide.

## 📄 License

Educational project for academic purposes.

---

**Status:** 🚧 In Development
