# Custom AI Training App

Train small language models with custom data for educational purposes.

## 🎯 Purpose

This application enables teachers and educators to:
- Create custom AI assistants trained on specific curriculum
- Generate training data using OpenRouter or local Ollama models
- Fine-tune small language models (1B-7B parameters) efficiently
- Export models for deployment in educational environments

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- (Optional) [Ollama](https://ollama.ai) - for generating SFT training data locally
- (Optional) NVIDIA GPU with CUDA for faster training
- (Optional) Apple Silicon Mac for MPS acceleration
- Internet connection - for downloading Hugging Face models

### Installation

1. Clone this repository (private access required)

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** The first installation may take 10-20 minutes as it downloads PyTorch and Unsloth dependencies.

### Running the App

```bash
python app.py
```

The app will launch in your browser at `http://localhost:7860`

## 📋 Current Status

**Phase 1: Setup & Infrastructure** ✅ Complete
- ✅ Project structure
- ✅ Hardware detection (CUDA/MPS/CPU)
- ✅ Project management system
- ✅ Gradio UI with 4 tabs

**Phase 2: RAG Data Preparation** ✅ Complete
- ✅ Multi-format document upload (PDF, TXT, MD, DOCX, HTML, IPYNB, CSV, XLSX, JSON)
- ✅ Text chunking and SQLite storage
- ✅ Database import/merge/export (SQLite, JSON, JSONL)

**Phase 3: Model Selection** ✅ Complete
- ✅ Hugging Face model selection
- ✅ Model validation
- ✅ Recommended Unsloth-compatible models for education
- ✅ Output format configuration

**Phase 4: SFT Data Generation** ✅ Complete
- ✅ OpenRouter integration (GPT-4, Claude, Llama 3)
- ✅ Ollama local model support
- ✅ Custom prompt configuration
- ✅ Async generation with progress tracking
- ✅ JSONL export for training

**Phase 5: Training & Export** ✅ Complete
- ✅ Unsloth-powered fine-tuning
- ✅ LoRA/QLoRA support
- ✅ Checkpoint management
- ✅ GGUF export (for Ollama/llama.cpp)
- ✅ Safetensors export (for HuggingFace)
- ✅ Direct Ollama import

**Phase 6: Settings & Polish** ✅ Complete
- ✅ Global settings management (settings.json)
- ✅ API key configuration (OpenRouter, Ollama)
- ✅ Customizable SFT prompts
- ✅ Training parameter defaults
- ✅ Export format defaults
- ✅ Hardware override options

## 🏗️ Project Structure

```
custom-ai/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
│
├── core/                     # Business logic
│   ├── hardware_detector.py # GPU/RAM detection
│   ├── project_manager.py   # Project CRUD operations
│   ├── rag_processor.py     # RAG data processing (Phase 2)
│   ├── ollama_client.py     # Ollama integration (Phase 3)
│   ├── sft_generator.py     # SFT data generation (Phase 4)
│   ├── trainer.py           # Training logic (Phase 5)
│   └── exporter.py          # Model export (Phase 5)
│
├── ui/                       # Gradio UI components
│   ├── tab_rag.py           # RAG data preparation
│   ├── tab_model.py         # Model selection
│   ├── tab_sft.py           # SFT data generation
│   ├── tab_training.py      # Training & export
│   └── settings.py          # Global settings (Phase 6)
│
├── utils/                    # Helper functions
│   ├── config.py            # Global config management (Phase 6)
│   ├── logger.py
│   └── validators.py
│
├── projects/                 # User projects (created at runtime)
│   └── [project_name]/
│       ├── config.json
│       ├── data/
│       ├── models/
│       └── logs/
│
└── docs/                     # Documentation
    ├── c-ai_prd.md
    ├── implementation_plan.md
    └── setup_guide.md
```

## 💻 Hardware Requirements

### Minimum (CPU-only)
- CPU: 4 cores, 2.5GHz+
- RAM: 16GB
- Storage: 20GB free
- ⚠️ Training will be 50-100x slower than GPU

### Recommended (GPU)
- GPU: NVIDIA RTX 3060 (12GB VRAM) or Apple M1/M2 (16GB unified)
- RAM: 16GB
- Storage: 50GB free
- ⏱️ 7B model fine-tuning in 1-3 hours

### Optimal (High-End)
- GPU: NVIDIA RTX 4090 (24GB) or Apple M2 Ultra (64GB+)
- RAM: 32GB+
- Storage: 100GB+ SSD
- ⚡ 13B model fine-tuning in 2-4 hours

## 🔒 License

Proprietary - All Rights Reserved

This software is closed source and intended for private use only.

## 📧 Support

For issues or questions, contact the development team.

---

**Built with:** Python • Gradio • Unsloth • PyTorch
# custom-ai
