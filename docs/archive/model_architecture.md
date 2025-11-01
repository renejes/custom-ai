# Model Architecture & Workflow

## Two Types of Models in the App

This app uses **two different types of models** for different purposes:

### 1. Base Model for Training (Hugging Face)
**Location:** Tab 2 - Model Selection
**Purpose:** This is the model that will be fine-tuned
**Source:** Hugging Face

**Examples:**
- `unsloth/llama-3-8b-Instruct-bnb-4bit`
- `unsloth/Phi-3-mini-4k-instruct`
- `unsloth/Qwen2-1.5B-Instruct`

**Why Hugging Face?**
- Unsloth requires Hugging Face format models
- Direct access to model weights
- Efficient fine-tuning with LoRA/QLoRA

### 2. Helper Model for SFT Data Generation (Ollama or OpenRouter)
**Location:** Tab 3 - SFT Data Generation
**Purpose:** Generate training examples (questions & answers)
**Source:** Ollama (local) OR OpenRouter (cloud)

**Ollama Examples:**
- `llama3:8b` (local)
- `phi3:mini` (local)
- `qwen2:7b` (local)

**OpenRouter Examples:**
- `openai/gpt-4-turbo`
- `anthropic/claude-3-sonnet`
- `meta-llama/llama-3-70b-instruct`

**Why Ollama/OpenRouter?**
- Generate diverse training data
- No need to download - just inference
- Can use more powerful models for data generation

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. TAB 1: RAG Data Preparation                             │
│    Upload textbooks/documents → Chunk → SQLite DB          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TAB 2: Select Base Model (for Training)                 │
│    Choose Hugging Face model:                              │
│    • unsloth/llama-3-8b-Instruct-bnb-4bit                  │
│    This model will be fine-tuned!                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TAB 3: Generate SFT Training Data                       │
│    Use Helper Model (Ollama/OpenRouter):                   │
│    • Llama 3 (via Ollama) OR GPT-4 (via OpenRouter)       │
│    → Generates Q&A pairs → Saves to JSONL                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. TAB 4: Train the Base Model                             │
│    Load: Base Model (from Tab 2)                           │
│    Data: SFT Training Data (from Tab 3)                    │
│    → Fine-tune with Unsloth                                │
│    → Export as GGUF/Safetensors                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Separation?

### Training Model (Hugging Face)
- **Needs:** Full model weights for training
- **Downloads:** ~1-5GB model files
- **Process:** Fine-tuning with gradient updates
- **Output:** Customized model

### Helper Model (Ollama/OpenRouter)
- **Needs:** Just API access for inference
- **Downloads:** Nothing (Ollama) or API calls (OpenRouter)
- **Process:** Generate text based on prompts
- **Output:** Training data (JSONL)

---

## Example Scenario

**Goal:** Create a 5th-grade math tutor

### Step 1: Prepare Knowledge (Tab 1)
- Upload 5th-grade math textbooks
- Upload practice problems
- System chunks and indexes content

### Step 2: Choose Training Model (Tab 2)
```
Base Model: unsloth/Phi-3-mini-4k-instruct
(This will become your custom math tutor)
```

### Step 3: Generate Training Data (Tab 3)
```
Helper Model: llama3:8b (via Ollama)

Prompt: "Generate 5th-grade math questions and answers"

Output (JSONL):
{"instruction": "What is 25 × 4?", "output": "25 × 4 = 100"}
{"instruction": "Solve: 3/4 + 1/2", "output": "3/4 + 1/2 = 1 1/4"}
...
```

### Step 4: Train (Tab 4)
```
Base Model: unsloth/Phi-3-mini-4k-instruct
Training Data: 500 Q&A pairs (from Tab 3)
Training Time: ~30 minutes

Result: custom-math-tutor-5th-grade.gguf
```

---

## Model Compatibility

### ✅ Compatible for Training (Hugging Face)
- Models with "unsloth/" prefix (optimized)
- Models with "bnb-4bit" (quantized for efficiency)
- Any Hugging Face causal LM (but slower)

### ✅ Compatible for SFT Generation (Ollama)
- Any Ollama model you have installed
- Run `ollama list` to see available models

### ✅ Compatible for SFT Generation (OpenRouter)
- GPT-4, Claude, Llama, Mixtral, etc.
- Requires API key from openrouter.ai

---

## FAQ

**Q: Can I use the same model for training and SFT generation?**
A: Technically yes, but not recommended. Training models are optimized for efficiency (4-bit), while helper models for generation should be full-precision for better quality.

**Q: Do I need Ollama installed?**
A: Only if you want to generate SFT data locally. You can use OpenRouter instead (cloud-based).

**Q: Can I skip Tab 3 and provide my own training data?**
A: Yes! Just place a JSONL file in `projects/your-project/data/sft_data.jsonl` with the format:
```json
{"instruction": "question", "output": "answer"}
```

**Q: What if I don't have a GPU?**
A: Training will work on CPU but be very slow (hours/days). Consider using Google Colab or a cloud GPU for training.
