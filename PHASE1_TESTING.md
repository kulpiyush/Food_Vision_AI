# Phase 1 Testing Guide - Basic UI

## ✅ What's Been Created

### 1. **Main Application** (`app.py`)
- Streamlit UI with image upload
- Results display area
- Sidebar with settings
- Placeholder analysis functionality

### 2. **Utility Functions**
- `utils/image_processing.py` - Image preprocessing functions
- `utils/nutrition_calculator.py` - Nutrition lookup functions
- `models/vision_model.py` - Vision model wrapper (placeholder)

### 3. **Nutrition Database**
- `data/nutrition_db.csv` - Sample Indian food nutrition data (15 foods)

## 🚀 How to Test Phase 1

### Step 1: Install Dependencies
```bash
# Make sure you're in the project directory
cd /Users/piyushkulkarni/Documents/Automated_Nutritional_Analysis_App

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

The app should open in your browser at `http://localhost:8501`

### Step 3: Test Image Upload
1. **Upload an image:**
   - Click "Choose an image of Indian food..."
   - Select any food image (JPG, PNG, JPEG)
   - The image should display in the left column

2. **Click "Analyze Food" button:**
   - You should see a spinner "Analyzing food image..."
   - Results should appear in the right column

3. **Check Results:**
   - ✅ Detected food name (random from placeholder)
   - ✅ Confidence score
   - ✅ Top 3 predictions
   - ✅ Nutritional information (if food is in database)
   - ℹ️ Placeholder messages for AI description and Q&A

## 📋 Expected Behavior

### ✅ What Should Work:
- Image upload and display
- Image validation (size, format)
- Placeholder food prediction
- Nutrition lookup from CSV database
- Results display with metrics
- Sidebar settings (UI only, not functional yet)

### ⚠️ What's Placeholder:
- Food detection (random selection, not real AI)
- Model inference (will be real in Phase 2)
- AI description (coming in Phase 2)
- Q&A interface (coming in Phase 2)

## 🧪 Test Cases

### Test Case 1: Valid Image Upload
1. Upload a valid food image (JPG/PNG)
2. **Expected:** Image displays, "Analyze Food" button appears
3. **Result:** ✅ Should work

### Test Case 2: Food Analysis
1. Upload image and click "Analyze Food"
2. **Expected:** 
   - Random Indian food name appears
   - Confidence score (85-95%)
   - Top 3 predictions shown
   - Nutrition data if food is in database
3. **Result:** ✅ Should work (with placeholder data)

### Test Case 3: Nutrition Lookup
1. Analyze an image
2. If detected food is in database (Biryani, Dosa, etc.), nutrition should show
3. **Expected:** Calories, Fat, Carbs, Protein displayed
4. **Result:** ✅ Should work for foods in sample database

### Test Case 4: Invalid Image
1. Try uploading a very small image (< 32x32)
2. **Expected:** Error message displayed
3. **Result:** ✅ Should show validation error

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Streamlit not found"
**Solution:**
```bash
pip install streamlit
```

### Issue: "Nutrition database not found"
**Solution:**
```bash
# The database should be created automatically, but if missing:
python -c "from utils.nutrition_calculator import create_sample_nutrition_db; create_sample_nutrition_db()"
```

### Issue: App won't start
**Solution:**
- Check if port 8501 is already in use
- Try: `streamlit run app.py --server.port 8502`

## 📊 Sample Nutrition Database

The sample database includes these Indian foods:
- Biryani, Dosa, Idli, Samosa, Curry
- Naan, Roti, Dal, Paneer Tikka, Butter Chicken
- Palak Paneer, Chole, Rajma, Aloo Gobi, Baingan Bharta

**Note:** This is sample data. You should replace with real nutritional values in Phase 2.

## ✅ Phase 1 Completion Checklist

- [x] Streamlit app runs without errors
- [x] Image upload works
- [x] Image displays correctly
- [x] "Analyze Food" button works
- [x] Placeholder predictions appear
- [x] Nutrition data displays (for foods in database)
- [x] UI looks good and is responsive
- [x] No console errors

## 🎯 Next Steps (Phase 2)

Once Phase 1 is working:
1. Fine-tune vision model on Indian cuisine
2. Replace placeholder predictions with real model inference
3. Integrate Generative AI (Ollama)
4. Add AI descriptions and Q&A

## 💡 Tips

- **Test with different images:** Try various food images
- **Check console:** Look for any errors in terminal
- **Test nutrition lookup:** Try foods both in and out of database
- **UI testing:** Check on different screen sizes (responsive design)

---

**Phase 1 Status:** ✅ Ready for Testing

Run `streamlit run app.py` and test it out! 🚀

