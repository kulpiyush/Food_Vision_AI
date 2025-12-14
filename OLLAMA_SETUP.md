# Ollama Setup Guide

## What is Ollama?

Ollama is a tool that runs large language models (LLMs) locally on your machine. It's free, open-source, and doesn't require API keys.

## Installation

### macOS

1. **Download Ollama:**
   - Visit: https://ollama.ai
   - Click "Download for macOS"
   - Open the downloaded file and follow the installation instructions

2. **Verify Installation:**
   ```bash
   ollama --version
   ```

3. **Pull Llama 3.2 Model:**
   ```bash
   ollama pull llama3.2
   ```
   This downloads the model (takes a few minutes, ~2GB)

4. **Test Ollama:**
   ```bash
   ollama run llama3.2 "Hello, how are you?"
   ```

### Linux

1. **Install Ollama:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Model:**
   ```bash
   ollama pull llama3.2
   ```

3. **Test:**
   ```bash
   ollama run llama3.2 "Hello"
   ```

### Windows

1. **Download from:** https://ollama.ai
2. **Install the .exe file**
3. **Open PowerShell/CMD and run:**
   ```bash
   ollama pull llama3.2
   ```

## Alternative Models

If you want to use a different model:

```bash
# Smaller, faster model (less accurate)
ollama pull llama3.2:1b

# Larger, more accurate model (slower)
ollama pull llama3.2:3b

# Mistral (alternative to Llama)
ollama pull mistral

# Phi-3 (Microsoft's model)
ollama pull phi3
```

Then update `config/config.yaml`:
```yaml
genai:
  model_name: "mistral"  # or "phi3", etc.
```

## Verifying Setup

Run this to test if Ollama is working:

```bash
python test_genai.py
```

You should see:
```
Available: True
```

## Troubleshooting

### Issue: "Ollama not found"
**Solution:** Make sure Ollama is installed and in your PATH. Restart your terminal after installation.

### Issue: "Connection refused"
**Solution:** Make sure Ollama service is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama (it usually starts automatically)
# On macOS: Open Ollama app
# On Linux: systemctl start ollama
```

### Issue: "Model not found"
**Solution:** Pull the model:
```bash
ollama pull llama3.2
```

### Issue: Slow responses
**Solution:** 
- Use a smaller model: `ollama pull llama3.2:1b`
- Or use GPU if available (Ollama will use it automatically)

## Using Without Ollama

The app will work without Ollama, but:
- ❌ Food descriptions won't be generated
- ❌ Q&A interface won't work
- ✅ Vision model predictions will still work
- ✅ Nutrition data will still be displayed

You can install Ollama later to enable GenAI features.

## Next Steps

Once Ollama is set up:
1. Run `python test_genai.py` to verify
2. Run `streamlit run app.py` to use GenAI features
3. Upload a food image and see AI-generated descriptions!

---

**Note:** Ollama runs locally, so:
- ✅ No internet required (after initial download)
- ✅ No API keys needed
- ✅ Free to use
- ✅ Privacy-friendly (data stays on your machine)

