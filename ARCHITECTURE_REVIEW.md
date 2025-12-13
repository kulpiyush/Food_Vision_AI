# Architecture Review Guide - FoodVisionAI

## 🎯 Purpose
This document helps you understand and review the system architecture before implementation.

---

## 📐 System Overview

FoodVisionAI is a **3-layer architecture**:

```
┌─────────────────────────────────────────┐
│   LAYER 1: USER INTERFACE (Streamlit)   │
│   - Image upload & display              │
│   - Results visualization               │
│   - Q&A chat interface                  │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│   LAYER 2: APPLICATION LOGIC            │
│   - Image preprocessing                 │
│   - Pipeline orchestration              │
│   - Error handling                      │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│   LAYER 3: AI MODULES + DATA            │
│   - Vision Model (EfficientNet-B0)      │
│   - Generative AI (Ollama/Llama)        │
│   - Nutritional Database                │
└─────────────────────────────────────────┘
```

---

## 🔄 Complete Data Flow (Step-by-Step)

### Step 1: User Uploads Image
- **Where:** Streamlit UI (`app.py`)
- **What happens:** User selects/upload image file
- **Output:** Image file in memory

### Step 2: Image Preprocessing
- **Where:** `utils/image_processing.py`
- **What happens:**
  - Resize to 224x224 (EfficientNet input size)
  - Normalize pixel values
  - Convert to tensor format
- **Output:** Preprocessed image tensor

### Step 3: Food Detection/Classification
- **Where:** `models/vision_model.py`
- **What happens:**
  - Load fine-tuned EfficientNet-B0 model
  - Run inference on preprocessed image
  - Get top predictions (food name + confidence)
- **Output:** 
  ```python
  {
    "food_name": "Biryani",
    "confidence": 0.92,
    "alternative_predictions": [...]
  }
  ```

### Step 4: Nutritional Lookup
- **Where:** `utils/nutrition_calculator.py`
- **What happens:**
  - Search `nutrition_db.csv` for detected food
  - Retrieve nutritional values (per 100g)
  - Calculate based on portion size
- **Output:**
  ```python
  {
    "calories": 350,
    "fat_g": 12.5,
    "carbs_g": 45.0,
    "protein_g": 15.0,
    "fiber_g": 3.0
  }
  ```

### Step 5: Generative AI Processing
- **Where:** `models/genai_model.py`
- **What happens:**
  - Send prompt to Ollama (Llama 3.2)
  - Generate food description
  - Generate nutritional analysis
  - Generate meal suggestions
- **Output:**
  ```python
  {
    "description": "Biryani is a fragrant rice dish...",
    "nutrition_analysis": "This meal provides...",
    "suggestions": "Consider pairing with..."
  }
  ```

### Step 6: Display Results
- **Where:** Streamlit UI (`app.py`)
- **What happens:**
  - Show detected food with confidence
  - Display nutritional breakdown (charts/tables)
  - Show AI-generated descriptions
  - Enable Q&A chat
- **Output:** User sees complete analysis

---

## 🧩 Component Details

### 1. Vision Model Component

**File:** `models/vision_model.py`

**Responsibilities:**
- Load pretrained EfficientNet-B0
- Fine-tune on Indian cuisine dataset
- Run inference on new images
- Return predictions with confidence scores

**Key Functions:**
```python
class VisionModel:
    def __init__(self, model_path, num_classes=20)
    def load_model(self)
    def predict(self, image_tensor)
    def get_top_predictions(self, image_tensor, top_k=5)
```

**For Indian Cuisine:**
- Classes: ~15-20 Indian dishes
- Training data: 50-100 images per dish
- Fine-tuning: Last few layers only (transfer learning)

**Model Flexibility:**
- Easy to swap EfficientNet-B0 → EfficientNet-B2 → ResNet-50
- Same interface, different model loading

---

### 2. Generative AI Component

**File:** `models/genai_model.py`

**Responsibilities:**
- Connect to Ollama (local) or API (fallback)
- Generate food descriptions
- Provide nutritional insights
- Answer user questions

**Key Functions:**
```python
class GenAIModel:
    def __init__(self, provider="ollama", model="llama3.2")
    def generate_description(self, food_name, nutrition_data)
    def analyze_nutrition(self, food_name, nutrition_data)
    def suggest_alternatives(self, food_name, nutrition_data)
    def answer_question(self, question, context)
```

**Prompt Engineering:**
- Templates for each use case
- Include context (food name, nutrition)
- Optimize for Indian cuisine knowledge

**Fallback Strategy:**
- Try Ollama first (local, free)
- If fails, use OpenAI/Anthropic API
- Graceful degradation

---

### 3. Nutritional Database

**File:** `data/nutrition_db.csv`

**Structure:**
```csv
food_name,calories,fat_g,carbs_g,protein_g,fiber_g,per_100g
Biryani,350,12.5,45.0,15.0,3.0,100
Dosa,150,5.0,25.0,4.0,2.0,100
...
```

**Responsibilities:**
- Store nutritional values for Indian foods
- Support lookup by food name
- Handle variations (e.g., "Chicken Biryani" vs "Biryani")

**Data Sources:**
- USDA FoodData Central
- Indian food nutrition databases
- Custom research for local dishes

---

### 4. UI Component (Streamlit)

**File:** `app.py`

**Layout:**
```
┌─────────────────────────────────────┐
│  FoodVisionAI - Nutritional Analysis │
├─────────────────────────────────────┤
│  [Upload Image]                     │
│  [Image Preview]                    │
│  [Analyze Button]                   │
├─────────────────────────────────────┤
│  Results:                             │
│  - Detected Food: Biryani (92%)    │
│  - Nutrition Chart                  │
│  - AI Description                   │
│  - Q&A Chat                         │
└─────────────────────────────────────┘
```

**Features:**
- Image upload widget
- Real-time analysis
- Interactive charts
- Chat interface for questions

---

## 🔗 Component Interactions

### Interaction 1: Image → Vision Model
```
app.py (UI)
  ↓ uploads image
utils/image_processing.py
  ↓ preprocesses
models/vision_model.py
  ↓ returns prediction
app.py (UI)
  ↓ displays food name
```

### Interaction 2: Vision Model → Nutrition DB
```
models/vision_model.py
  ↓ returns "Biryani"
utils/nutrition_calculator.py
  ↓ queries nutrition_db.csv
  ↓ returns nutrition data
app.py (UI)
  ↓ displays nutrition chart
```

### Interaction 3: Nutrition → GenAI
```
utils/nutrition_calculator.py
  ↓ provides nutrition data
models/genai_model.py
  ↓ generates description
app.py (UI)
  ↓ displays AI text
```

---

## 🎨 Architecture Patterns Used

### 1. **Modular Design**
- Each component in separate file
- Easy to test and modify independently
- Clear separation of concerns

### 2. **Pipeline Pattern**
- Data flows through stages
- Each stage transforms input → output
- Easy to add/remove stages

### 3. **Strategy Pattern** (for models)
- Easy to swap vision models
- Easy to swap GenAI providers
- Configuration-driven

### 4. **Facade Pattern** (UI)
- Streamlit UI simplifies complex backend
- User doesn't see internal complexity
- Clean interface

---

## 🔍 Key Design Decisions

### Why EfficientNet-B0?
- ✅ Small model size (~5MB)
- ✅ Fast inference (~50ms on CPU)
- ✅ Good accuracy for food classification
- ✅ Easy to fine-tune
- ✅ Can upgrade to B2 if needed

### Why Ollama (Local)?
- ✅ No API costs during development
- ✅ Privacy (data stays local)
- ✅ No internet required
- ✅ Can test extensively
- ✅ API fallback available

### Why Streamlit?
- ✅ Fast to build (hours, not days)
- ✅ Python-only (no frontend skills needed)
- ✅ Built-in widgets (upload, charts, chat)
- ✅ Good for demos
- ✅ Easy to deploy

### Why CSV for Nutrition DB?
- ✅ Simple to create/edit
- ✅ No database setup needed
- ✅ Easy to version control
- ✅ Can upgrade to SQLite later

---

## 🚀 Scalability Considerations

### Current Design (MVP):
- Single food detection
- Local processing
- CSV database
- Basic UI

### Future Scalability:
- **Multi-food:** Add YOLO for detection
- **Cloud:** Deploy models to cloud
- **Database:** Migrate to SQLite/PostgreSQL
- **API:** Convert to REST API
- **Mobile:** Build mobile app with same backend

---

## ⚠️ Potential Challenges & Solutions

### Challenge 1: Model Accuracy
**Problem:** EfficientNet-B0 might not be accurate enough  
**Solution:** 
- Try EfficientNet-B2 (better accuracy)
- Try ResNet-50 (proven for food)
- Improve training data quality

### Challenge 2: Ollama Not Working
**Problem:** Ollama installation issues or slow  
**Solution:**
- Use OpenAI API as fallback
- Test Ollama early
- Have API keys ready

### Challenge 3: Limited Training Data
**Problem:** Not enough Indian food images  
**Solution:**
- Use data augmentation (rotate, flip, color)
- Use Food-101 dataset (has some Indian foods)
- Collect custom images
- Use transfer learning (needs less data)

### Challenge 4: Nutrition Data Missing
**Problem:** Some Indian foods not in database  
**Solution:**
- Research and add manually
- Use USDA database as base
- Estimate from similar foods
- Allow user to add custom entries

---

## ✅ Architecture Checklist

Before starting implementation, verify:

- [x] Architecture is clear and documented
- [x] Component responsibilities defined
- [x] Data flow understood
- [x] Technology choices made
- [x] Fallback strategies planned
- [x] Scalability considered
- [x] Challenges identified

---

## 📚 Next Steps After Review

1. **Understand the flow:** Read through data flow section
2. **Review components:** Understand each component's role
3. **Check decisions:** Verify decisions align with architecture
4. **Ask questions:** Clarify anything unclear
5. **Start coding:** Begin with Phase 1 (basic UI)

---

## 💡 Questions to Consider

1. **Do you understand the data flow?**
   - Image → Preprocess → Vision → Nutrition → GenAI → Display

2. **Are the components clear?**
   - Vision model, GenAI, Nutrition DB, UI

3. **Do the technology choices make sense?**
   - EfficientNet-B0, Ollama, Streamlit

4. **Are fallbacks planned?**
   - Model alternatives, API fallback

5. **Ready to start implementation?**
   - If yes, proceed to Phase 1!

---

**If everything looks good, you're ready to start building! 🚀**

