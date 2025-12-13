# Project Decisions Log

## ✅ Confirmed Decisions

### 1. Local Cuisine Focus
**Decision:** **Indian Cuisine**
- Focus on popular Indian dishes (Biryani, Dosa, Curry, Naan, etc.)
- Training dataset will include Indian food images
- Nutritional database will prioritize Indian food items

### 2. Vision Model Strategy
**Decision:** **Start with EfficientNet-B0, flexible to try others**
- **Primary:** EfficientNet-B0 (lightweight, fast, good accuracy)
- **Backup options if needed:**
  - EfficientNet-B2 (more accurate, slightly slower)
  - ResNet-50 (proven, reliable)
  - Vision Transformer (ViT) (state-of-the-art, needs more data)
  - CLIP (good for zero-shot, but larger)
- **Approach:** Start simple, iterate based on performance

### 3. Generative AI
**Decision:** **Ollama (local) with API fallback**
- **Primary:** Ollama + Llama 3.2 (free, local, privacy-friendly)
- **Fallback:** OpenAI GPT-4o-mini or Anthropic Claude (for better quality if needed)
- **Benefits:** No API costs during development, can switch to API for demo

### 4. Extended Features Priority
**Decision:** **Core features first, extended features later**
- **Phase 1-2:** Focus on core requirements
- **Phase 3:** Add extended features if time permits
- **Rationale:** Better to have a polished core than incomplete extended features

## 📋 Architecture Implications

### For Indian Cuisine:
- **Dataset needs:** 10-20 common Indian dishes
  - Biryani, Dosa, Idli, Samosa, Curry, Naan, Roti, Dal, Paneer dishes, etc.
- **Nutritional database:** Should include Indian food items with accurate values
- **Model considerations:** Indian food has diverse textures and presentations, may need good data augmentation

### For Model Flexibility:
- **Code structure:** Make model loading modular (easy to swap)
- **Evaluation:** Test EfficientNet-B0 first, benchmark before trying others
- **Time management:** Don't spend too long on model selection, pick one and optimize

### For Ollama Setup:
- **Local setup:** Need to install Ollama and pull Llama 3.2 model
- **Fallback ready:** Keep API integration code ready but commented
- **Testing:** Test Ollama early to ensure it works on your system

## 🎯 Next Steps Based on Decisions

1. **Update config.yaml** with Indian cuisine settings
2. **Research Indian food datasets** (Food-101 has some, may need custom)
3. **Prepare Indian food nutritional database**
4. **Set up Ollama** early to test GenAI integration
5. **Create modular model loader** for easy model swapping

