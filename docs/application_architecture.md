# Application Architecture

## Overview

Custom AI Training App is a Gradio-based application for training small language models with RAG (Retrieval-Augmented Generation) data. Designed for educational purposes, it allows users to create custom LLMs by uploading documents, generating training data, and fine-tuning models.

**Platform:** macOS with Apple Silicon (MPS support)
**Python Version:** 3.11
**UI Framework:** Gradio 4.0+
**Training Framework:** transformers + PEFT (LoRA/QLoRA)

---

## Project Structure

```
custom-ai/
├── app.py                     # Main Gradio application entry point
├── requirements.txt           # Python dependencies
├── pyrightconfig.json         # Type checking configuration
├── .vscode/
│   └── settings.json          # VSCode workspace settings (Pylance)
├── core/
│   ├── hardware_detector.py   # Detect GPU/CPU/MPS capabilities
│   ├── rag_processor.py       # Document processing & RAG pipeline (semantic chunking)
│   ├── sft_generator.py       # Generate SFT training data via LLM
│   ├── trainer.py             # Model training (transformers+PEFT)
│   ├── exporter.py            # Export models (Safetensors)
│   ├── ollama_client.py       # Ollama API client
│   └── project_manager.py     # Project creation & management
├── ui/
│   ├── tab_project.py         # Tab 0: Project Management (NEW)
│   ├── tab_rag.py             # Tab 1: RAG Data Preparation (multi-database + viewer)
│   ├── tab_cpt.py             # Tab 2: CPT Training (local model selection)
│   ├── tab_sft.py             # Tab 3: SFT Data Generation (DataFrame chunk selection)
│   ├── tab_training.py        # Tab 4: SFT Training
│   └── settings.py            # Tab 5: Global Settings
├── utils/
│   └── config.py              # Global configuration management (multi-database registry)
├── projects/                  # User projects directory (custom locations supported)
│   └── {project_name}/
│       ├── config.json        # Project-specific config
│       ├── data/
│       │   ├── databases/     # Multiple named RAG databases (NEW)
│       │   │   ├── wissensbasis.db
│       │   │   └── {custom}.db
│       │   └── sft_data.jsonl # Generated training data
│       ├── models/
│       │   ├── cpt_model/     # CPT trained model
│       │   ├── sft_model/     # SFT trained model
│       │   ├── checkpoints/   # Training checkpoints
│       │   └── final/         # Exported models
│       └── logs/              # Training logs
└── docs/
    ├── application_architecture.md  # This file
    ├── session_summary.md           # Development session log
    ├── workflow_4phase.md           # 4-phase workflow documentation
    └── c-ai_prd.md                  # Product requirements
```

---

## Core Components

### 1. app.py - Main Application

**Purpose:** Initialize Gradio interface and coordinate all tabs

**Key Functions:**
- `create_project_selector()`: Project creation/selection UI
- `main()`: Build 5-tab Gradio interface
- Coordinates state between tabs via Gradio State

**Dependencies:** All UI modules, utils.config

---

### 2. core/hardware_detector.py

**Purpose:** Detect available hardware (NVIDIA GPU, AMD GPU, Apple MPS, CPU)

**Key Functions:**
```python
def detect_hardware() -> Dict[str, Any]:
    """Returns dict with device, vram, ram, cpu info"""

def get_recommended_settings(hardware: Dict) -> Dict:
    """Returns recommended batch_size, use_4bit, etc."""
```

**Logic:**
1. Check for CUDA (NVIDIA)
2. Check for ROCm (AMD)
3. Check for MPS (Apple Silicon)
4. Fall back to CPU
5. Detect VRAM/RAM/CPU cores
6. Return recommendations

**Output Example:**
```python
{
    "device": "mps",
    "device_name": "Apple M1 Max",
    "vram_gb": 0,  # MPS shares system RAM
    "ram_gb": 64,
    "cpu_cores": 10,
    "recommended_batch_size": 4,
    "recommended_use_4bit": False  # MPS doesn't support 4-bit
}
```

---

### 3. core/rag_processor.py

**Purpose:** Process documents into RAG-ready chunks and store in SQLite

**Key Classes:**
- `RAGProcessor`: Main document processing class

**Key Functions:**
```python
def __init__(
    self,
    db_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    use_semantic_chunking: bool = True,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    """Initialize with semantic or fixed chunking"""

def process_files(
    self,
    files: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Tuple[bool, str]:
    """Process uploaded files into RAG database"""

def merge_databases(
    self,
    source_db_paths: List[str],
    target_db_path: str
) -> Tuple[bool, str, int]:
    """Merge multiple databases into a target database"""
```

**Supported File Types:**
- PDF (.pdf)
- Word (.docx)
- Text (.txt, .md)
- HTML (.html)
- Jupyter Notebooks (.ipynb)
- Excel (.xlsx)
- CSV (.csv)

**Chunking Strategies:**

1. **Semantic Chunking** (Default, NEW):
   - Uses LangChain `SemanticChunker`
   - Multilingual embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
   - 118M parameters, supports 50+ languages including German
   - Splits text based on semantic similarity, not fixed size
   - Better context preservation for Q&A generation
   - Alternative: `google/embeddinggemma-300m` (100+ languages)

2. **Fixed-Size Chunking** (Backward compatible):
   - Uses `RecursiveCharacterTextSplitter`
   - Fixed character/token count
   - Configurable overlap

**Pipeline:**
1. Load file with appropriate loader (PyPDF, python-docx, etc.)
2. Split into chunks (Semantic or Fixed)
3. Store chunks in SQLite with metadata
4. Index for fast retrieval

**Database Schema:**
```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding BLOB  -- Optional for semantic search
);
```

---

### 4. core/sft_generator.py

**Purpose:** Generate SFT (Supervised Fine-Tuning) training data using LLM

**Key Classes:**
- `SFTGenerator`: Orchestrates data generation

**Key Functions:**
```python
def generate_sft_data(
    self,
    rag_db_path: Path,
    model_name: str,
    model_type: str,  # "ollama" or "openrouter"
    num_examples: int,
    system_prompt: str,
    user_prompt_template: str
) -> Tuple[bool, str]:
    """Generate Q&A pairs from RAG data"""
```

**Process:**
1. Load chunks from RAG database
2. For each chunk:
   - Build prompt with system_prompt + user_prompt_template
   - Send to Ollama or OpenRouter API
   - Parse response (expect JSON: `{"instruction": "...", "output": "..."}`)
3. Save all Q&A pairs to `sft_data.jsonl`

**API Support:**
- **Ollama:** Local inference via HTTP API (http://localhost:11434)
- **OpenRouter:** Cloud inference via REST API (requires API key)

**Output Format (JSONL):**
```json
{"instruction": "What is machine learning?", "output": "Machine learning is..."}
{"instruction": "Explain neural networks", "output": "Neural networks are..."}
```

---

### 5. core/trainer.py (MPS-Compatible)

**Purpose:** Train models using transformers + PEFT (LoRA/QLoRA)

**Important:** This version uses standard HuggingFace libraries instead of Unsloth because **Unsloth does not support Apple Silicon MPS**.

**Key Classes:**
- `ModelTrainer`: Manages model loading, training, and saving

**Key Functions:**

```python
def load_model(
    self,
    model_id: str,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False
) -> Tuple[bool, str]:
    """Load base model for MPS (4-bit not supported on MPS)"""
```

```python
def prepare_lora(
    self,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05
) -> Tuple[bool, str]:
    """Apply LoRA adapter to model using PEFT"""
```

```python
def train(
    self,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_checkpointing: bool = True,
    progress_callback: Optional[Callable] = None
) -> Tuple[bool, str]:
    """Train model with LoRA using transformers.Trainer"""
```

**Device Selection Logic:**
```python
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32
```

**MPS Limitations:**
- No 4-bit quantization (uses FP16 instead)
- No Flash Attention 2
- Slower than CUDA/Unsloth but functional

**Training Pipeline:**
1. Load base model (AutoModelForCausalLM)
2. Load tokenizer (AutoTokenizer)
3. Apply LoRA config (peft.LoraConfig)
4. Load SFT dataset (datasets.load_dataset)
5. Train with Trainer (transformers.Trainer)
6. Save adapter weights

---

### 6. core/exporter.py (MPS-Compatible)

**Purpose:** Export trained models to various formats

**Supported Formats:**
- ✅ **Safetensors:** Full model + adapter weights (recommended)
- ❌ **GGUF:** Requires Unsloth (not available on MPS)
- ❌ **Ollama:** Requires GGUF conversion (not available)

**Key Functions:**

```python
def export_to_safetensors(
    self,
    model_path: Path,
    output_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """Export model to Safetensors format"""
```

**Why Safetensors Only?**
- GGUF export requires Unsloth's `save_model_gguf()` method
- Unsloth doesn't work on Apple Silicon
- Safetensors is compatible with all HuggingFace tools

**Workaround for GGUF:**
If you need GGUF format:
1. Export to Safetensors on Mac
2. Transfer to Linux/Windows with NVIDIA GPU
3. Use llama.cpp or Unsloth to convert to GGUF

---

### 7. ui/tab_project.py (NEW)

**Purpose:** Tab 0 - Project Management

**UI Components:**
- Project name input
- Project description textarea
- Custom location selector with browse button
- Create project button
- Active project selector dropdown
- Project folder structure viewer
- Delete project button with confirmation
- Open in Finder/Explorer button

**Workflow:**
1. User creates new project with optional custom location
2. User selects active project from dropdown
3. User views organized folder structure
4. User can open project folder or delete project

**Folder Structure Created:**
```
Project/
├── data/
│   ├── databases/         # All RAG databases
│   │   ├── wissensbasis.db
│   │   └── [custom].db
│   └── sft_data.jsonl
├── models/
│   ├── cpt_model/         # CPT trained model
│   ├── sft_model/         # SFT trained model
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Exported models
└── logs/
```

---

### 8. ui/tab_rag.py

**Purpose:** Tab 1 - RAG Data Preparation (Multi-Database Support)

**UI Components:**
- Database name input for creating new databases
- Active database selector dropdown
- Database import (.db file upload)
- Database merge (select multiple → merge)
- File upload (multiple files)
- Semantic chunking toggle
- Chunk size/overlap sliders
- Process button
- **Database Viewer** with pagination (NEW):
  - Load chunks button
  - Previous/Next 1000 buttons
  - Go to page input with jump button
  - Page info display (Page X of Y, Chunks A-B of Total)
  - DataFrame showing: ID, Filename, Chunk Index, Preview
- Status display

**Multi-Database Features (NEW):**
- Create multiple named databases (e.g., "math", "physics", "history")
- Import existing .db files as new databases
- Merge multiple databases into one
- Switch between databases with active selection
- Each database stored in `data/databases/{name}.db`

**Semantic Chunking Features (NEW):**
- Toggle between Semantic and Fixed chunking
- Multilingual embedding model selection
- Max chunk size (1000 chars default)
- Chunk overlap (100 chars default)
- First-run downloads embedding model (~120MB)

**Database Viewer with Pagination (NEW):**
- Displays total chunk count
- Loads 1000 chunks per page (SQL LIMIT/OFFSET)
- Navigation: Previous/Next 1000, Direct page jump
- Shows current range and total (e.g., "Chunks 1001-2000 of 5432")
- Performance optimized for large databases (10K+ chunks)

**Workflow:**
1. User creates or selects database
2. User uploads documents (PDF, DOCX, TXT, etc.)
3. User configures semantic chunking (optional)
4. User adjusts chunk size/overlap
5. Click "Process Files"
6. RAGProcessor processes files into selected database
7. User views chunks with pagination
8. Optional: Import or merge databases

---

### 9. ui/tab_cpt.py

**Purpose:** Tab 2 - CPT (Continued Pre-Training) Training

**UI Components:**
- **Model Path Input** (free-form text, NEW):
  - Supports local paths: `/Users/you/models/Phi-3-mini`
  - Supports HuggingFace IDs: `microsoft/Phi-3-mini-4k-instruct`
- **Browse Button** (NEW): Opens folder selection dialog
- **Verify Button** (NEW): Checks if model exists and is valid
- Active database display (loads chunks from here)
- Training parameters (epochs, batch_size, learning_rate, lora_rank)
- Gradient checkpointing toggle
- Train CPT button
- Progress bar
- Status display

**Model Selection (NEW):**
- Users place models anywhere on disk or use HuggingFace IDs
- Recommended models dropdown for quick selection
- Verify checks for `config.json` and `*.safetensors` files
- No auto-download UI (removed)

**Workflow:**
1. User enters or browses for model path
2. User verifies model exists
3. User loads RAG chunks from active database
4. User configures training parameters
5. Click "Start CPT Training"
6. ModelTrainer trains on raw chunks (language modeling)
7. CPT model saved to `models/cpt_model/`

---

### 10. ui/tab_sft.py

**Purpose:** Tab 3 - SFT Data Generation

**UI Components:**
- **Load Chunks Button**: Load all chunks from active database
- **DataFrame Chunk Selection** (NEW):
  - Checkboxes for selecting chunks
  - Columns: Select, ID, Source, Chunk Index, Preview
  - Select All / Deselect All buttons
  - Shows "X / Y selected" counter
- Model type selector (Ollama / OpenRouter)
- Model name input
- System prompt textbox
- User prompt template textbox
- Generate button
- Preview generated data

**Chunk Selection (NEW):**
- User browses all chunks in DataFrame table
- Multi-select with checkboxes
- Only selected chunks used for Q&A generation
- Replaces previous slider-based selection

**Workflow:**
1. User clicks "Load Chunks from Active Database"
2. User selects specific chunks to use (DataFrame checkboxes)
3. User selects Ollama or OpenRouter
4. User enters model name (e.g., "llama3:8b" for Ollama)
5. User customizes prompts (optional)
6. Click "Generate SFT Data"
7. SFTGenerator queries LLM with selected RAG chunks
8. Saves output to `sft_data.jsonl`

---

### 11. ui/tab_training.py

**Purpose:** Tab 4 - SFT Training

**UI Components:**
- CPT model path display (auto-loaded from Tab 2)
- Training parameters (epochs, batch_size, learning_rate, lora_rank)
- Gradient checkpointing toggle
- Train SFT button
- Progress bar
- Export format selector (Safetensors only)
- Export button

**Workflow:**
1. User adjusts training hyperparameters
2. Click "Start SFT Training"
3. ModelTrainer loads CPT model from Tab 2
4. Applies LoRA adapter
5. Trains on `sft_data.jsonl` from Tab 3
6. Training progress shown in real-time
7. SFT model saved to `models/sft_model/`
8. After training, user selects export format
9. Click "Export Model"
10. Model saved to `models/final/`

---

### 12. ui/settings.py

**Purpose:** Tab 5 - Global Settings

**UI Components:**
- API Configuration (OpenRouter API key, Ollama base URL)
- SFT Prompts (default system prompt, user prompt template)
- Training Defaults (learning_rate, epochs, batch_size, lora_rank)
- Export Defaults (format, quantization)
- Hardware Overrides (force CPU, max RAM)

**Workflow:**
1. User configures global defaults
2. Click "Save Settings"
3. Settings saved to `settings.json` in root directory
4. All tabs use these defaults

---

### 13. utils/config.py

**Purpose:** Global configuration management (singleton pattern)

**Key Functions:**
```python
def get_global_config() -> GlobalConfig:
    """Get or create singleton config instance"""

class GlobalConfig:
    def get_api_config(self) -> Dict:
        """Return OpenRouter/Ollama settings"""

    def get_sft_prompts(self) -> Dict:
        """Return default SFT prompts"""

    def get_training_defaults(self) -> Dict:
        """Return default training hyperparameters"""

    def save(self) -> bool:
        """Save config to JSON file"""

    # Multi-Database Management (NEW)
    def get_rag_databases(self) -> Dict[str, str]:
        """Get all RAG databases (name → path mapping)"""

    def add_rag_database(self, name: str, path: str) -> bool:
        """Add a new RAG database"""

    def get_active_rag_database(self) -> str:
        """Get active RAG database name"""

    def set_active_rag_database(self, name: str) -> bool:
        """Set active RAG database"""

    # Local Model Management (NEW)
    def get_local_models_dir(self) -> str:
        """Get local models directory"""

    def scan_local_models(self) -> List[str]:
        """Scan local models directory and return list of valid models"""
```

**Config Structure:**
```python
{
    "api": {
        "openrouter_api_key": "",
        "ollama_base_url": "http://localhost:11434"
    },
    "sft_prompts": {
        "system_prompt": "You are a helpful assistant...",
        "user_prompt_template": "Generate a Q&A pair about: {chunk}"
    },
    "training_defaults": {
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 4,
        "lora_rank": 16,
        "use_4bit": False,
        "gradient_checkpointing": True
    },
    "export_defaults": {
        "format": "safetensors",
        "quantization": "none"
    },
    # Multi-Database Configuration (NEW)
    "rag_databases": {
        "wissensbasis": "projects/my_project/data/databases/wissensbasis.db",
        "math": "projects/my_project/data/databases/math.db"
    },
    "active_rag_database": "wissensbasis",
    # Local Model Management (NEW)
    "local_models_dir": "models",
    "available_local_models": ["Phi-3-mini", "Llama-3.2-1B"]
}
```

---

### 14. core/project_manager.py (MOVED from utils/)

**Purpose:** Create and manage user projects with custom locations

**Key Functions:**
```python
class Project:
    def __init__(
        self,
        name: str,
        base_path: str = "projects",
        custom_location: Optional[str] = None
    ):
        """Initialize project with optional custom location"""

    def initialize(self):
        """Create organized directory structure"""

    def get_folder_structure(self) -> str:
        """Return formatted folder tree"""

def create_project(
    name: str,
    description: str = "",
    custom_location: Optional[str] = None
) -> Tuple[bool, str]:
    """Create new project with custom location support"""

def list_projects() -> List[str]:
    """List all existing projects"""

def delete_project(name: str) -> Tuple[bool, str]:
    """Delete project with confirmation"""

def open_project_folder(name: str) -> Tuple[bool, str]:
    """Open project folder in file manager"""
```

**Project Structure Created:**
```
projects/{project_name}/  # Or custom location
├── config.json
├── data/
│   ├── databases/         # Multiple named databases (NEW)
│   │   ├── wissensbasis.db
│   │   └── {custom}.db
│   └── sft_data.jsonl
├── models/
│   ├── cpt_model/         # CPT trained model (NEW)
│   ├── sft_model/         # SFT trained model (NEW)
│   ├── checkpoints/       # Training checkpoints (NEW)
│   └── final/             # Exported models (NEW)
└── logs/                  # Training logs (NEW)
```

**Custom Location Support (NEW):**
- Users can place projects anywhere on disk
- Default: `projects/` directory in app root
- Custom: Any user-specified path (e.g., `/Users/me/Desktop/MyProjects/`)
- Project metadata stored in `projects.json` in app root

---

## Data Flow

### Complete 4-Phase Training Pipeline (UPDATED)

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 0: Project Setup (Tab 0 - NEW)                            │
│    User creates project with custom location                    │
│    → ProjectManager creates folder structure                    │
│    → databases/, cpt_model/, sft_model/, checkpoints/, logs/    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: RAG Data Preparation (Tab 1)                           │
│    User creates/selects database (multi-database support)       │
│    → User uploads PDF/DOCX/TXT                                  │
│    → RAGProcessor with semantic chunking                        │
│    → Saves to data/databases/{db_name}.db                       │
│    → User views chunks with pagination (1000 per page)          │
│    → Optional: Import/merge databases                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: CPT Training (Tab 2 - NEW)                             │
│    User enters/browses for base model path                      │
│    → ModelTrainer loads base model (local or HF ID)             │
│    → Loads raw chunks from active database                      │
│    → Trains on chunks (language modeling)                       │
│    → Saves CPT model to models/cpt_model/                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: SFT Data Generation (Tab 3)                            │
│    User loads chunks in DataFrame table                         │
│    → User selects specific chunks (checkboxes)                  │
│    → SFTGenerator reads selected chunks                         │
│    → For each chunk: query Ollama/OpenRouter                    │
│    → Saves Q&A pairs to data/sft_data.jsonl                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 4: SFT Training & Export (Tab 4)                          │
│    ModelTrainer loads CPT model from Phase 2                    │
│    → Applies LoRA adapter (PEFT)                                │
│    → Trains on sft_data.jsonl from Phase 3                      │
│    → Saves SFT model to models/sft_model/                       │
│    → User exports to Safetensors                                │
│    → Saved to models/final/safetensors_{timestamp}/             │
└──────────────────────────────────────────────────────────────────┘
```

### Multi-Database Architecture (NEW)

```
Project/data/databases/
├── wissensbasis.db      # Default knowledge base
├── math.db              # Math-specific database
├── physics.db           # Physics-specific database
└── history.db           # History-specific database

User Actions:
1. Create new database → New .db file created
2. Import database → Copy external .db file
3. Merge databases → Combine multiple .db files into one
4. Select active database → Used for CPT training and SFT generation
5. View database → Load chunks with pagination (1000 per page)
```

---

## API Integrations

### Ollama API

**Endpoint:** `http://localhost:11434/api/generate`

**Request:**
```json
{
    "model": "llama3:8b",
    "prompt": "Generate a Q&A pair about...",
    "stream": false
}
```

**Response:**
```json
{
    "response": "{\"instruction\": \"...\", \"output\": \"...\"}"
}
```

### OpenRouter API

**Endpoint:** `https://openrouter.ai/api/v1/chat/completions`

**Headers:**
```
Authorization: Bearer {api_key}
Content-Type: application/json
```

**Request:**
```json
{
    "model": "openai/gpt-4-turbo",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
}
```

**Response:**
```json
{
    "choices": [
        {
            "message": {
                "content": "{\"instruction\": \"...\", \"output\": \"...\"}"
            }
        }
    ]
}
```

---

## Configuration Files

### .vscode/settings.json

**Purpose:** Enable workspace-wide Pylance error checking

**Key Settings:**
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv311/bin/python",
    "python.analysis.diagnosticMode": "workspace",  // Show all errors
    "python.analysis.typeCheckingMode": "basic"
}
```

### pyrightconfig.json

**Purpose:** Project-wide type checking configuration

```json
{
    "include": ["app.py", "core", "ui", "utils"],
    "exclude": ["venv", "venv311"],
    "pythonVersion": "3.11",
    "typeCheckingMode": "basic"
}
```

---

## Dependencies (requirements.txt)

### UI Framework
- `gradio>=4.0.0` - Web UI

### PyTorch (Apple Silicon MPS Support)
- `torch>=2.1.0`
- `torchvision>=0.16.0`
- `torchaudio>=2.1.0`

### HuggingFace Libraries
- `transformers>=4.40.0` - Model loading/training
- `datasets>=2.18.0` - Dataset handling
- `peft>=0.10.0` - LoRA/QLoRA adapters
- `trl>=0.8.0` - SFT training utilities
- `accelerate>=0.28.0` - Multi-GPU/device support
- `bitsandbytes>=0.42.0` - Quantization (limited on macOS)

### RAG & Document Processing
- `langchain>=0.1.0` - RAG orchestration
- `langchain-text-splitters>=0.0.1` - Text chunking
- `langchain-experimental>=0.0.50` - SemanticChunker (NEW)
- `langchain-huggingface>=0.0.1` - HF embeddings integration (NEW)
- `sentence-transformers>=2.3.0` - Embedding models for semantic chunking (NEW)
- `pypdf>=3.17.0` - PDF parsing
- `python-docx>=1.1.0` - Word document parsing
- `beautifulsoup4>=4.12.0` - HTML parsing
- `nbformat>=5.9.0` - Jupyter notebook parsing
- `openpyxl>=3.1.0` - Excel parsing
- `lxml>=4.9.0` - XML parsing

### API Clients
- `aiohttp>=3.9.0` - Async HTTP (OpenRouter)
- `requests>=2.31.0` - HTTP (Ollama)

### Utilities
- `python-dotenv>=1.0.0` - Environment variables
- `tqdm>=4.66.0` - Progress bars
- `psutil>=5.9.0` - System info

---

## Hardware Requirements

### Minimum
- **CPU:** Apple Silicon (M1/M2/M3) or x86_64
- **RAM:** 16GB
- **Storage:** 10GB free

### Recommended
- **CPU:** Apple M1 Max/Ultra or NVIDIA RTX 3090+
- **RAM:** 32GB+
- **Storage:** 50GB+ SSD

### GPU Support
- ✅ **Apple MPS** (M1/M2/M3): Fully supported
- ✅ **NVIDIA CUDA** (Linux/Windows): Fully supported
- ✅ **AMD ROCm** (Linux): Supported
- ✅ **CPU Only:** Supported (very slow)

---

## Platform-Specific Notes

### macOS (Apple Silicon)

**What Works:**
- Training with transformers + PEFT
- MPS acceleration (faster than CPU)
- Safetensors export
- All RAG/SFT features

**Limitations:**
- No 4-bit quantization (uses FP16)
- No Unsloth optimizations
- No GGUF export
- Slower than CUDA/Unsloth

**Workaround:**
Train on Mac → Export Safetensors → Convert to GGUF on Linux/Windows

### Linux/Windows (NVIDIA GPU)

**What Works:**
- Everything (full Unsloth support if you switch back to original trainer.py)
- 4-bit quantization
- GGUF export
- Fastest training

**To Enable Unsloth:**
1. Use backup files: `trainer_unsloth.py.backup` and `exporter_unsloth.py.backup`
2. Install: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
3. Restore original trainer/exporter logic

---

## Type System

All core modules use Python 3.11+ type hints with `from __future__ import annotations`.

**Example:**
```python
from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Callable

def train(
    self,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 3,
    progress_callback: Optional[Callable] = None
) -> Tuple[bool, str]:
    """Type hints enable IDE autocomplete and Pylance checking"""
    pass
```

---

## Error Handling

All core functions return `Tuple[bool, str]`:
- `(True, "Success message")` on success
- `(False, "Error message")` on failure

**Example:**
```python
success, message = trainer.load_model("meta-llama/Llama-3.2-1B")
if not success:
    gr.Warning(f"Failed to load model: {message}")
    return
```

---

## Security Considerations

1. **API Keys:** Stored in project config.json (not committed to git)
2. **File Uploads:** Validated file extensions before processing
3. **Model Loading:** Only load from trusted HuggingFace repos
4. **SQLite Injection:** Parameterized queries only
5. **Path Traversal:** All paths validated with `Path.resolve()`

---

## Testing Strategy

### Manual Testing Workflow
1. Create new project
2. Upload sample PDF
3. Generate SFT data with Ollama (llama3:8b)
4. Download small model (Qwen2-1.5B-Instruct)
5. Train for 1 epoch
6. Export to Safetensors
7. Verify export files exist

### Unit Tests (Future)
- `test_rag_processor.py`: Test document chunking
- `test_sft_generator.py`: Test API calls (mocked)
- `test_trainer.py`: Test model loading (small model)
- `test_exporter.py`: Test Safetensors export

---

## Future Enhancements

### Planned Features
- ✅ Add GGUF export support for Linux/Windows
- ✅ Add quantization options (2-bit, 3-bit, 8-bit)
- ✅ Add model evaluation metrics (perplexity, accuracy)
- ✅ Add chat interface to test trained models
- ✅ Add multi-GPU training support
- ✅ Add resume training from checkpoint
- ✅ Add LoRA merging UI

### Known Issues
- Unsloth not supported on Apple Silicon
- 4-bit quantization not available on MPS
- GGUF export requires Unsloth (not available on MPS)
- No progress streaming during SFT generation (async needed)

---

## Troubleshooting

### Issue: "Unsloth only works on NVIDIA, AMD and Intel GPUs"
**Solution:** Use the MPS-compatible trainer.py (already implemented)

### Issue: Training very slow on MPS
**Solution:**
1. Reduce batch_size to 2-4
2. Enable gradient_checkpointing
3. Use smaller model (Qwen2-1.5B instead of Llama-3-8B)

### Issue: Out of memory during training
**Solution:**
1. Reduce batch_size
2. Reduce max_seq_length
3. Enable gradient_checkpointing
4. Use smaller LoRA rank (8 instead of 16)

### Issue: GGUF export not available
**Solution:** This is expected on macOS. Use Safetensors and convert on Linux/Windows.

### Issue: Ollama connection failed
**Solution:**
1. Check Ollama is running: `ollama list`
2. Verify base URL: `http://localhost:11434`
3. Test with: `curl http://localhost:11434/api/tags`

---

## License & Credits

**Framework:** Built with Gradio, HuggingFace Transformers, PEFT
**Inspiration:** Unsloth (not used on macOS but conceptually inspired)
**Document Processing:** LangChain
**Model Hosting:** HuggingFace Hub

---

---

## New Features and Enhancements (2025-11-01)

### 1. Semantic Chunking

**Implementation:**
- LangChain `SemanticChunker` with multilingual embeddings
- Default model: `paraphrase-multilingual-MiniLM-L12-v2` (118M params, 50+ languages)
- Alternative: `google/embeddinggemma-300m` (100+ languages, 300M params)

**How It Works:**
1. Text is split into sentences
2. Each sentence is embedded using multilingual model
3. Embeddings are compared for semantic similarity
4. Chunks are created at points where similarity drops (topic changes)
5. Max chunk size (1000 chars) as fallback for very long sections

**Benefits:**
- Better context preservation for German educational content
- Semantically coherent chunks ideal for Q&A generation
- Avoids mid-sentence or mid-thought splits

**Performance:**
- First run: 1-2 minutes (downloads 120MB model)
- Subsequent runs: 5-10 seconds per 100 pages
- Memory: ~500MB RAM during chunking

**Configuration:**
- Toggle: Semantic vs Fixed chunking
- Max chunk size: 1000 characters (default)
- Chunk overlap: 100 characters (default)
- Embedding model: Configurable (default multilingual)

---

### 2. Multi-Database System

**Architecture:**
- Multiple named SQLite databases per project
- Each database stored in `data/databases/{name}.db`
- Active database selection for training/generation
- Database registry in `settings.json`

**Operations:**
- **Create:** New empty database with custom name
- **Import:** Copy external .db file as new database
- **Merge:** Combine multiple databases into one
- **Select:** Switch active database for workflows
- **View:** Browse chunks with pagination

**Use Cases:**
- Separate databases for different subjects (math, physics, history)
- Import pre-built knowledge bases
- Merge related content before training
- Test different chunking strategies

---

### 3. Database Viewer with Pagination

**Implementation:**
- SQL LIMIT/OFFSET for efficient querying
- 1000 chunks per page (configurable)
- DataFrame display with ID, Filename, Chunk Index, Preview

**Navigation:**
- **Previous/Next Buttons:** Navigate by 1000 chunks
- **Page Input:** Jump directly to specific page
- **Page Info:** "Page X of Y (Chunks A-B of Total)"

**Performance:**
- Handles databases with 10K+ chunks
- Only loads 1000 chunks at a time
- Query time: <1 second per page

**SQL Query:**
```sql
SELECT id, filename, chunk_index, chunk_text
FROM documents
ORDER BY filename, chunk_index
LIMIT 1000 OFFSET {page_offset}
```

---

### 4. Project Management with Custom Locations

**Features:**
- Create projects anywhere on disk
- Default: `projects/` in app root
- Custom: User-specified path (e.g., Desktop, external drive)
- Project metadata in `projects.json`

**Folder Structure:**
```
Project/
├── data/
│   ├── databases/         # Multiple named databases
│   └── sft_data.jsonl
├── models/
│   ├── cpt_model/         # CPT trained model
│   ├── sft_model/         # SFT trained model
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Exported models
└── logs/                  # Training logs
```

**Actions:**
- Create new project
- Select active project
- View folder structure
- Open in file manager
- Delete with confirmation

---

### 5. Local Model Management

**Changes:**
- Removed HuggingFace auto-download UI
- Free-form model path input
- Supports local paths and HF IDs

**Model Path Examples:**
- Local: `/Users/me/models/Phi-3-mini`
- HuggingFace: `microsoft/Phi-3-mini-4k-instruct`

**Features:**
- **Recommended Models Dropdown:** Quick selection of common models
- **Verify Button:** Checks for `config.json` and `*.safetensors`
- **Flexible:** Users manage models themselves

**Rationale:**
- Users often download models manually
- More control over model storage location
- Simpler UI without download complexity

---

### 6. DataFrame Chunk Selection for SFT

**Implementation:**
- Replaced slider with DataFrame table
- Checkboxes for multi-select
- Shows: Select, ID, Source, Chunk Index, Preview

**Features:**
- Load all chunks from active database
- Select specific chunks with checkboxes
- Select All / Deselect All buttons
- Counter: "X / Y selected"
- Only selected chunks used for Q&A generation

**Benefits:**
- User sees actual chunk content before selection
- More precise control over training data
- Better for quality over quantity approach

---

## Contact & Support

For issues, feature requests, or questions, please refer to the project documentation or create an issue in the repository.
