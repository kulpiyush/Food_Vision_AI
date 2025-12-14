# Phase 2 Implementation Guide - Core AI Features

## 🎯 Phase 2 Goals

1. **Replace placeholder predictions with real vision model inference**
2. **Integrate Generative AI (Ollama/Llama) for descriptions and Q&A**
3. **Update app to use real models instead of placeholders**

## 📋 Step-by-Step Implementation Plan

### **Step 1: Set Up Real Vision Model Inference** ⏱️ ~30 minutes

**Goal:** Replace placeholder predictions with actual EfficientNet inference

#### 1.1 Update `models/vision_model.py`
- [ ] Load pretrained EfficientNet-B0 (ImageNet weights)
- [ ] Create proper class mapping for Indian foods
- [ ] Implement real inference function
- [ ] Handle model loading and caching

#### 1.2 Create Class Names Mapping
- [ ] Define list of Indian food classes (15-20 foods)
- [ ] Map model outputs to food names
- [ ] Store in config or separate file

#### 1.3 Test Vision Model
- [ ] Test with sample images
- [ ] Verify predictions work correctly
- [ ] Check inference speed

---

### **Step 2: Create Generative AI Module** ⏱️ ~45 minutes

**Goal:** Create GenAI wrapper for Ollama/Llama integration

#### 2.1 Create `models/genai_model.py`
- [ ] Create GenAIModel class
- [ ] Implement Ollama integration
- [ ] Add functions for:
  - Food description generation
  - Nutritional analysis
  - Meal suggestions
  - Q&A handling

#### 2.2 Set Up Ollama (if not already installed)
- [ ] Install Ollama: https://ollama.ai
- [ ] Pull Llama 3.2 model: `ollama pull llama3.2`
- [ ] Test Ollama connection

#### 2.3 Create Prompt Templates
- [ ] Food description prompt
- [ ] Nutritional analysis prompt
- [ ] Q&A prompt template

---

### **Step 3: Integrate Real Models into App** ⏱️ ~30 minutes

**Goal:** Update `app.py` to use real models instead of placeholders

#### 3.1 Update Vision Model Integration
- [ ] Replace `create_placeholder_prediction()` with real model
- [ ] Load model on app startup (with caching)
- [ ] Update prediction flow

#### 3.2 Add GenAI Integration
- [ ] Initialize GenAI model
- [ ] Add food description generation
- [ ] Add Q&A chat interface
- [ ] Handle errors gracefully

#### 3.3 Update UI
- [ ] Replace placeholder messages with real GenAI outputs
- [ ] Add Q&A chat interface
- [ ] Add loading states for GenAI responses

---

### **Step 4: Testing & Refinement** ⏱️ ~30 minutes

**Goal:** Test everything works and fix any issues

#### 4.1 Test Vision Model
- [ ] Test with various food images
- [ ] Verify predictions are reasonable
- [ ] Check error handling

#### 4.2 Test GenAI
- [ ] Test food descriptions
- [ ] Test Q&A interface
- [ ] Verify responses are relevant

#### 4.3 End-to-End Testing
- [ ] Upload image → Get prediction → See nutrition → Get AI description → Ask questions
- [ ] Test error cases (no internet, Ollama down, etc.)
- [ ] Performance testing

---

## 🚀 Detailed Implementation Steps

### **STEP 1: Real Vision Model Inference**

#### Task 1.1: Update Vision Model Class

**File:** `models/vision_model.py`

**Changes needed:**
1. Add Indian food class names list
2. Implement proper model loading
3. Update predict() to return real predictions
4. Add model initialization function

#### Task 1.2: Create Class Names File (Optional)

**File:** `config/class_names.yaml` or hardcode in vision_model.py

**Content:** List of 15-20 Indian food names matching your nutrition database

---

### **STEP 2: Generative AI Module**

#### Task 2.1: Create GenAI Model File

**File:** `models/genai_model.py` (NEW FILE)

**Features to implement:**
- `GenAIModel` class
- `generate_food_description(food_name, nutrition_data)` method
- `analyze_nutrition(nutrition_data)` method
- `answer_question(question, context)` method
- Ollama client integration

#### Task 2.2: Install and Set Up Ollama

**Commands:**
```bash
# Install Ollama (visit https://ollama.ai for macOS)
# Or use: brew install ollama

# Pull Llama 3.2 model
ollama pull llama3.2

# Test it works
ollama run llama3.2 "Hello, how are you?"
```

#### Task 2.3: Add Ollama to Requirements

**File:** `requirements.txt`

**Add:**
```
ollama>=0.1.0
```

---

### **STEP 3: App Integration**

#### Task 3.1: Update app.py - Vision Model

**Changes:**
1. Import real VisionModel class
2. Initialize model on startup (with caching)
3. Replace placeholder prediction with real inference
4. Handle model loading errors

#### Task 3.2: Update app.py - GenAI

**Changes:**
1. Import GenAIModel
2. Initialize GenAI on startup
3. Add food description section (replace placeholder)
4. Add Q&A chat interface
5. Handle GenAI errors gracefully

#### Task 3.3: Update UI Components

**Changes:**
1. Replace "AI description will appear here" with real GenAI output
2. Add Q&A chat interface (text input + chat history)
3. Add loading indicators for GenAI responses
4. Style improvements

---

### **STEP 4: Testing**

#### Test Checklist:
- [ ] Vision model loads correctly
- [ ] Vision model makes predictions
- [ ] Predictions are reasonable (not random)
- [ ] Ollama connection works
- [ ] Food descriptions are generated
- [ ] Q&A interface works
- [ ] Error handling works (Ollama down, etc.)
- [ ] App doesn't crash on errors
- [ ] Performance is acceptable (< 5 seconds for inference)

---

## 📝 Implementation Order

**Recommended sequence:**

1. **First:** Set up real vision model (Step 1)
   - This gives you real predictions immediately
   - Test with images to verify it works

2. **Second:** Create GenAI module (Step 2)
   - Set up Ollama
   - Create genai_model.py
   - Test GenAI functions independently

3. **Third:** Integrate into app (Step 3)
   - Update app.py to use real vision model
   - Add GenAI to app
   - Update UI

4. **Fourth:** Test and refine (Step 4)
   - Fix any bugs
   - Improve prompts
   - Optimize performance

---

## 🛠️ Technical Details

### Vision Model Approach

**Option A: Use Pretrained Model (Fastest)**
- Load EfficientNet-B0 with ImageNet weights
- Use as-is (no fine-tuning)
- Map ImageNet classes to food (limited accuracy)
- **Pros:** Fast, no training needed
- **Cons:** Lower accuracy for food

**Option B: Use Food-Specific Pretrained Model (Recommended)**
- Use a model pretrained on food datasets (Food-101, etc.)
- Better accuracy for food classification
- Still no fine-tuning needed
- **Pros:** Better accuracy, still fast
- **Cons:** May not have Indian foods

**Option C: Fine-tune on Indian Cuisine (Best, but takes time)**
- Collect Indian food images
- Fine-tune EfficientNet on your dataset
- Best accuracy
- **Pros:** Best accuracy for Indian foods
- **Cons:** Requires dataset and training time

**For Phase 2, we'll use Option A or B (pretrained), then you can fine-tune later if needed.**

### GenAI Approach

**Primary:** Ollama with Llama 3.2
- Free, runs locally
- Good quality responses
- No API keys needed

**Fallback:** If Ollama doesn't work, we can add OpenAI/Anthropic later

---

## 📦 Files to Create/Modify

### New Files:
- [ ] `models/genai_model.py` - GenAI wrapper
- [ ] `config/class_names.yaml` (optional) - Food class names

### Files to Modify:
- [ ] `models/vision_model.py` - Real inference
- [ ] `app.py` - Integrate real models
- [ ] `requirements.txt` - Add ollama

---

## ✅ Success Criteria

Phase 2 is complete when:
- [x] Real vision model makes predictions (not random)
- [x] GenAI generates food descriptions
- [x] Q&A interface works
- [x] App runs without errors
- [x] All placeholder messages replaced with real outputs

---

## 🚨 Common Issues & Solutions

### Issue: Ollama not found
**Solution:** Install Ollama from https://ollama.ai

### Issue: Model predictions are wrong
**Solution:** This is expected with pretrained ImageNet model. Fine-tuning will improve accuracy.

### Issue: GenAI responses are slow
**Solution:** Normal for local LLM. Consider using smaller model or API fallback.

### Issue: Model loading is slow
**Solution:** Cache model in session state, only load once.

---

## 🎯 Next Steps After Phase 2

Once Phase 2 is complete:
- Fine-tune vision model on Indian cuisine (optional, for better accuracy)
- Add extended features (multi-food detection, portion estimation)
- Optimize performance
- Improve UI/UX

---

**Ready to start?** Let's begin with Step 1! 🚀

