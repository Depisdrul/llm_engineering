# Using Ollama for FREE Local AI Summarization

## Why Use Ollama?

- **100% FREE** - No API costs
- **Privacy** - All processing happens locally
- **Offline** - Works without internet
- **Good Quality** - Modern models are very capable for educational content

## Quick Start

### 1. Install Ollama

Visit [https://ollama.com](https://ollama.com) and download for Windows.

Or via command line:
```powershell
winget install Ollama.Ollama
```

### 2. Pull a Model

Choose one based on your hardware:

**Fast & Lightweight** (Recommended for most computers):
```powershell
ollama pull llama3.2
```

**Better Quality** (If you have 8GB+ RAM):
```powershell
ollama pull qwen2.5
```

**Best for Code** (7B model):
```powershell
ollama pull mistral
```

**Most Capable** (If you have 16GB+ RAM):
```powershell
ollama pull llama3.1:8b
```

### 3. Verify Ollama is Running

```powershell
ollama list
```

You should see your downloaded models.

### 4. Run Extraction with Ollama

```powershell
cd C:\Users\Pazzucconibt\REPO\llm_engineering

# Use default model (llama3.2)
python .knowledge-extraction\extract_all.py --use-llm --provider ollama

# Or specify a model
python .knowledge-extraction\extract_all.py --use-llm --provider ollama --model qwen2.5

# Extract specific weeks
python .knowledge-extraction\extract_all.py --use-llm --provider ollama --weeks 1,2,3
```

## Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2 | 3B | ⚡⚡⚡ | ⭐⭐⭐ | General use, fast |
| qwen2.5 | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Better reasoning |
| mistral | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Code & technical |
| llama3.1:8b | 8B | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| deepseek-coder | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Code examples |

## Performance Tips

### Speed Up Extraction

1. **Use a smaller model** - llama3.2 is 3x faster than llama3.1:8b
2. **Extract specific weeks** - Use `--weeks 1,2` instead of processing all
3. **GPU acceleration** - If you have an NVIDIA GPU, Ollama will use it automatically

### Memory Requirements

- **4GB RAM**: llama3.2 (3B)
- **8GB RAM**: qwen2.5, mistral (7B)
- **16GB RAM**: llama3.1:8b, dolphin-mixtral (8B)

### If Extraction is Slow

Don't worry! Even on slower hardware:
- Extract one week at a time: `--weeks 1`
- Run overnight for full extraction
- Quality is the same, just takes longer

## Troubleshooting

### "Connection refused" error

Ollama service isn't running. Start it:
```powershell
ollama serve
```

Or check if it's running:
```powershell
Get-Process ollama
```

### Model not found

Pull the model first:
```powershell
ollama pull llama3.2
```

### Out of memory

Use a smaller model:
```powershell
ollama pull llama3.2  # Only 3B parameters
python .knowledge-extraction\extract_all.py --use-llm --provider ollama --model llama3.2
```

## Comparing Results

Want to see the quality difference? Try both:

```powershell
# Extract Week 1 with local model
python .knowledge-extraction\extract_all.py --use-llm --provider ollama --weeks 1

# Compare with OpenAI (costs ~$0.50 for one week)
python .knowledge-extraction\extract_all.py --use-llm --provider openai --weeks 1
```

**Reality**: For educational content summarization, local models work very well! The quality difference is minimal for this use case.

## Recommended Workflow

1. **First extraction**: Use Ollama with llama3.2 (FREE, takes 1-2 hours)
2. **Review results**: Check quality of summaries
3. **If needed**: Re-run problem weeks with a larger model or OpenAI
4. **Ongoing**: Use Ollama for new weeks as you progress through course

## Advanced: Custom Models

You can use any model from [https://ollama.com/library](https://ollama.com/library):

```powershell
# Try different models
ollama pull phi3
ollama pull gemma2
ollama pull codellama

python .knowledge-extraction\extract_all.py --use-llm --provider ollama --model phi3
```

## Summary

**Bottom Line**: Start with Ollama + llama3.2. It's free, works offline, and quality is excellent for this use case. You can always try paid APIs later if you want even better summaries.

**Command to remember**:
```powershell
ollama pull llama3.2
python .knowledge-extraction\extract_all.py --use-llm --provider ollama
```

That's it! 🎉
