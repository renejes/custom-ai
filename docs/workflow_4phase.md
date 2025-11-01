# 4-Phase Training Pipeline

## Overview

This application implements a **4-phase training pipeline** for creating custom domain-specific language models:

1. **RAG Data** - Build knowledge base
2. **CPT Training** - Teach domain knowledge
3. **SFT Data** - Generate behavioral examples
4. **SFT Training** - Teach response behavior

**Key Principle:** `wissensbasis.sqlite` is the **Single Source of Truth** used in both Phase 2 (CPT) and Phase 3 (SFT Data).

---

## Phase 1: RAG Data Preparation

**Tab:** 1. RAG Data
**Goal:** Create knowledge base from documents
**Output:** `wissensbasis.sqlite`

### What Happens:
1. User uploads documents (PDF, DOCX, TXT, etc.)
2. Documents are chunked into smaller pieces
3. Chunks stored in SQLite database with metadata
4. This database becomes the Single Source of Truth

### Technical Details:
- **Chunking:** RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
- **Storage:** SQLite with full-text search
- **Schema:**
  ```sql
  CREATE TABLE rag_chunks (
      id INTEGER PRIMARY KEY,
      content TEXT,
      source TEXT,
      chunk_index INTEGER,
      metadata JSON
  );
  ```

### UI Features:
- File upload (multiple formats)
- Chunk size/overlap control
- Browse chunks in DataFrame
- Edit chunks before training

---

## Phase 2: CPT Training (Continued Pre-Training)

**Tab:** 2. CPT Training
**Goal:** Train model on domain knowledge
**Input:** `wissensbasis.sqlite` + Base Model
**Output:** CPT Model (with domain knowledge)

### What Happens:
1. Select base model from HuggingFace (e.g., Phi-3-mini, Qwen2)
2. Load chunks from `wissensbasis.sqlite`
3. Train model using **Language Modeling** (next-token prediction)
4. Model learns WHAT the knowledge is

### Why CPT?
- Model absorbs factual knowledge from documents
- Pre-training on domain-specific corpus
- Better foundation for later fine-tuning
- Model "reads" all the textbooks

### Technical Details:
- **Training Method:** Causal Language Modeling
- **Data Format:** Raw text chunks (no Q&A structure)
- **LoRA Rank:** 32 (higher for knowledge absorption)
- **Learning Rate:** 3e-4 (higher than SFT)
- **Epochs:** 3-5
- **Output:** `models/cpt_model/` (Safetensors + tokenizer)

### Training Parameters:
```python
{
    "learning_rate": 3e-4,  # Higher for CPT
    "epochs": 3,
    "batch_size": 4,
    "lora_rank": 32,  # Higher rank
    "max_seq_length": 2048
}
```

---

## Phase 3: SFT Data Generation

**Tab:** 3. SFT Data
**Goal:** Generate Q&A pairs from knowledge base
**Input:** `wissensbasis.sqlite` (same as Phase 2!)
**Output:** `sft_data.jsonl`

### What Happens:
1. Load chunks from `wissensbasis.sqlite` (same chunks as CPT!)
2. Browse chunks in UI
3. Use AI Assistant (Ollama/OpenRouter) to generate Q&A pairs
4. Generate in different styles:
   - Factual Q&A
   - Conversational
   - Socratic
   - Problem-Solving
   - Multiple Choice
5. Export as JSONL

### Why Generate from Same Data?
- **Consistency:** Model learns facts (Phase 2) then how to talk about them (Phase 4)
- **Single Source of Truth:** No data duplication or mismatch
- **Efficient:** One document upload, multiple uses

### Technical Details:
- **AI Models:** Ollama (local) or OpenRouter (cloud)
- **Sampling:** Even distribution across chunks
- **Format:** JSONL with `{"instruction": "...", "output": "..."}`
- **Styles:** Different prompts for different teaching approaches

### Example Output:
```json
{"instruction": "What is photosynthesis?", "output": "Photosynthesis is the process..."}
{"instruction": "Explain how plants make food", "output": "Plants use sunlight to..."}
```

### UI Features:
- Browse RAG chunks before generation
- Select AI provider (Ollama/OpenRouter)
- Choose Q&A style
- Configure prompts
- Preview generated data

---

## Phase 4: SFT Training (Supervised Fine-Tuning)

**Tab:** 4. SFT Training
**Goal:** Teach model HOW to respond
**Input:** CPT Model + `sft_data.jsonl`
**Output:** Final Model (knowledge + behavior)

### What Happens:
1. Load CPT model from Phase 2 (already has knowledge!)
2. Load Q&A pairs from `sft_data.jsonl`
3. Fine-tune with **LoRA** to teach response behavior
4. Model learns HOW to answer questions
5. Export final model

### Why Separate from CPT?
- **CPT:** Learns WHAT (facts, knowledge)
- **SFT:** Learns HOW (response style, behavior)
- More efficient than training both together
- Allows reusing CPT model for different behaviors

### Technical Details:
- **Training Method:** Supervised Fine-Tuning (Q&A format)
- **Base:** CPT model (not original base model!)
- **LoRA Rank:** 16 (smaller than CPT)
- **Learning Rate:** 2e-4 (lower than CPT)
- **Epochs:** 1-3 (fewer to avoid overfitting)
- **Output:** `models/final/` (CPT + SFT adapters merged)

### Training Parameters:
```python
{
    "learning_rate": 2e-4,  # Lower for SFT
    "epochs": 2,
    "batch_size": 4,
    "lora_rank": 16,  # Lower rank
    "gradient_checkpointing": True
}
```

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: RAG Data (Tab 1)                                      │
│                                                                 │
│ Upload: PDFs, DOCX, TXT                                        │
│    ↓                                                           │
│ Chunk: 500 tokens, 50 overlap                                 │
│    ↓                                                           │
│ Store: wissensbasis.sqlite                                     │
│                                                                 │
│ Output: SQLite database with 1000+ chunks                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: CPT Training (Tab 2)                                  │
│                                                                 │
│ Load: Phi-3-mini-4k-instruct (Base Model)                     │
│ Load: wissensbasis.sqlite chunks ← SINGLE SOURCE OF TRUTH     │
│    ↓                                                           │
│ Train: Language Modeling (next-token prediction)              │
│    Learning Rate: 3e-4                                         │
│    LoRA Rank: 32                                               │
│    Epochs: 3                                                    │
│    ↓                                                           │
│ Output: models/cpt_model/ (Model has KNOWLEDGE)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: SFT Data Generation (Tab 3)                           │
│                                                                 │
│ Load: wissensbasis.sqlite ← SAME DATA AS PHASE 2!             │
│ Browse: Chunks in UI                                           │
│    ↓                                                           │
│ Generate: With AI Assistant (Ollama/OpenRouter)               │
│    - Factual Q&A                                               │
│    - Conversational                                            │
│    - Socratic                                                  │
│    ↓                                                           │
│ Output: sft_data.jsonl (50-500 Q&A pairs)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: SFT Training (Tab 4)                                  │
│                                                                 │
│ Load: models/cpt_model/ (from Phase 2)                        │
│ Load: sft_data.jsonl (from Phase 3)                           │
│    ↓                                                           │
│ Train: Supervised Fine-Tuning (Q&A format)                    │
│    Learning Rate: 2e-4                                         │
│    LoRA Rank: 16                                               │
│    Epochs: 2                                                    │
│    ↓                                                           │
│ Output: models/final/ (Knowledge + Behavior)                  │
│ Export: Safetensors or GGUF                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Approach?

### 1. Separation of Concerns
- **CPT:** Domain knowledge (WHAT)
- **SFT:** Response behavior (HOW)
- Easier to debug and iterate

### 2. Single Source of Truth
- `wissensbasis.sqlite` used in Phase 2 AND 3
- No data duplication
- Guaranteed consistency

### 3. Efficiency
- Reuse CPT model for different SFT styles
- Can generate multiple SFT datasets without retraining CPT
- Example:
  ```
  Phase 2: CPT on biology textbook
  Phase 3a: Generate formal Q&A → formal_teacher.jsonl
  Phase 3b: Generate casual Q&A → casual_tutor.jsonl
  Phase 4a: SFT with formal_teacher.jsonl → formal model
  Phase 4b: SFT with casual_tutor.jsonl → casual model
  ```

### 4. Better Quality
- CPT gives strong knowledge foundation
- SFT shapes how knowledge is expressed
- Better than single-stage fine-tuning

---

## File Outputs

### After Phase 1:
```
projects/my-bio-assistant/
└── data/
    └── rag_db.sqlite  (wissensbasis.sqlite)
```

### After Phase 2:
```
projects/my-bio-assistant/
├── data/
│   ├── rag_db.sqlite
│   └── cpt_data.txt  (generated from SQLite)
└── models/
    └── cpt_model/
        ├── adapter_model.safetensors
        ├── adapter_config.json
        └── tokenizer files
```

### After Phase 3:
```
projects/my-bio-assistant/
├── data/
│   ├── rag_db.sqlite
│   ├── cpt_data.txt
│   └── sft_data.jsonl  (Q&A pairs)
└── models/
    └── cpt_model/
```

### After Phase 4:
```
projects/my-bio-assistant/
├── data/
│   ├── rag_db.sqlite
│   ├── cpt_data.txt
│   └── sft_data.jsonl
├── models/
│   ├── cpt_model/
│   ├── checkpoints/
│   └── final/  (merged CPT + SFT)
└── exports/
    └── safetensors_20250130/
        ├── model.safetensors
        ├── tokenizer files
        └── config.json
```

---

## Example Use Case: Biology Tutor

### Phase 1: Upload Data
- Upload 10 biology textbook PDFs
- Result: 2000 chunks in `wissensbasis.sqlite`

### Phase 2: CPT Training
- Base model: `microsoft/Phi-3-mini-4k-instruct`
- Train on 2000 biology chunks
- Duration: ~2 hours on MPS
- Result: `cpt_model` knows biology facts

### Phase 3: SFT Data
- Browse biology chunks
- Generate 100 Q&A pairs with `llama3:8b`
- Style: "Conversational tutor for 7th graders"
- Result: `sft_data.jsonl` with 100 pairs

### Phase 4: SFT Training
- Load `cpt_model` (has biology knowledge)
- Train on 100 Q&A pairs
- Duration: ~30 minutes on MPS
- Result: Final model that:
  - **Knows** biology (from CPT)
  - **Answers** like a 7th-grade tutor (from SFT)

### Test the Model:
```
User: "What is mitochondria?"
Model: "Great question! Mitochondria is like the power plant of the cell.
        It takes in nutrients and creates energy that the cell can use.
        Think of it like a tiny battery inside every cell!"
```

---

## Comparison: Traditional vs 4-Phase

### Traditional SFT Only:
```
Documents → Generate Q&A → Train model
```
**Problem:**
- Model must learn facts AND behavior simultaneously
- Less efficient
- Harder to iterate on behavior

### 4-Phase Pipeline:
```
Documents → CPT (learn facts) → Generate Q&A → SFT (learn behavior)
```
**Benefits:**
- Model learns facts first, behavior second
- Can reuse CPT model for different behaviors
- Higher quality final model
- Easier to debug

---

## Best Practices

### Phase 1 (RAG Data):
- Upload high-quality, relevant documents
- Use appropriate chunk size (500 for general text)
- Review chunks before proceeding

### Phase 2 (CPT):
- Choose smaller base models (1.5B-3B) for faster iteration
- Higher learning rate (3e-4 to 5e-4)
- More epochs (3-5) to absorb knowledge
- Monitor loss to avoid overfitting

### Phase 3 (SFT Data):
- Generate diverse Q&A pairs
- Match style to target audience
- Quality over quantity (50-100 good pairs > 500 bad pairs)
- Review generated data before training

### Phase 4 (SFT):
- Lower learning rate (2e-4 to 3e-4)
- Fewer epochs (1-3) to avoid overfitting
- Monitor validation loss
- Test model after training

---

## Troubleshooting

### Phase 2 CPT Training Slow:
- Use smaller base model (Qwen2-1.5B instead of Llama-3-8B)
- Reduce batch size
- Enable gradient checkpointing
- Use shorter max_seq_length (1024 instead of 2048)

### Phase 3 SFT Data Quality Low:
- Use better AI model (GPT-4 instead of GPT-3.5)
- Improve system prompt with examples
- Lower temperature (0.5-0.7 instead of 0.9)
- Review and manually edit generated pairs

### Phase 4 SFT Model Overfit:
- Reduce epochs (1-2 instead of 3-5)
- Lower learning rate (1e-4 instead of 2e-4)
- Add more diverse SFT data
- Use validation split to monitor

---

## Advanced: Multiple SFT Models

You can create multiple specialized models from one CPT model:

```
Phase 1: RAG Data (biology textbook)
Phase 2: CPT Training → bio_cpt_model

Phase 3a: Generate formal Q&A → formal.jsonl
Phase 4a: SFT Training → formal_bio_teacher

Phase 3b: Generate casual Q&A → casual.jsonl
Phase 4b: SFT Training → casual_bio_tutor

Phase 3c: Generate Socratic Q&A → socratic.jsonl
Phase 4c: SFT Training → socratic_bio_mentor
```

All three models have the same biology knowledge (from CPT) but different teaching styles (from different SFT data).

---

## Summary

The 4-Phase Training Pipeline provides:
1. ✅ **Clear separation** between knowledge (CPT) and behavior (SFT)
2. ✅ **Single Source of Truth** (`wissensbasis.sqlite`)
3. ✅ **Reusability** (one CPT model → multiple SFT models)
4. ✅ **Better quality** than single-stage fine-tuning
5. ✅ **Easier debugging** (identify if issue is knowledge or behavior)

**Key Principle:** The same raw data (`wissensbasis.sqlite`) is used for both CPT training (Phase 2) and SFT data generation (Phase 3), ensuring perfect consistency between what the model learned and how it's taught to respond.
