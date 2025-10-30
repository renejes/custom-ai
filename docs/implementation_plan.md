# Custom AI Training App - Finaler Implementierungsplan

## Projekt-Übersicht

**Zweck**: Desktop-Anwendung für Custom LLM Fine-Tuning mit RAG-Datenvorbereitung und SFT-Training

**Zielgruppe**: Forschung & Bildung (Studierende, Forscher, ML-Praktiker)

**Lizenzmodell**: Closed Source, private Distribution

---

## Tech Stack

### Core Technologies
- **UI Framework**: Gradio (Browser-basiert)
- **Training**: Unsloth (inkludiert automatisch Transformers + PEFT für LoRA/QLoRA)
- **Model Integration**: Ollama (lokale Modelle)
- **SFT Data Generation**: OpenRouter API (Cloud) + Ollama API (lokal)
- **Database**: SQLite (RAG-Daten)
- **Language**: Python 3.10+

### GPU Support
- **CUDA**: NVIDIA GPUs (Windows/Linux)
- **MPS**: Apple Silicon (macOS)
- **CPU Fallback**: Für Systeme ohne GPU (mit Performance-Warnung)

---

## Entwicklungs-Strategie

### Phase 1: Entwicklung (JETZT)
- **Pure Python** entwickeln
- **Privates Git Repository**
- Entwicklung auf macOS (MPS)
- Schnelle Iteration ohne Distribution-Overhead

### Phase 2: Distribution (SPÄTER)
Mehrere Distributionswege anbieten:

1. **PyInstaller Executable** (.exe für Windows, .app für macOS)
   - Kein Python-Installation nötig
   - Code nicht einsehbar
   - Standalone-Binary

2. **Docker Image** (private Registry)
   - Für Workshops/Enterprise
   - Konsistente Umgebung
   - CUDA-Support

3. **Python Package** (für Power-User/Early Access)
   - Mit Lizenzschlüssel-System
   - Requirements.txt + Setup-Skript

### Code-Schutz (Phase 2)
- PyArmor für Code-Obfuscation
- License Key System
- Private Download-Portal (nur nach Registrierung/Kauf)

---

## Architektur

### Projektstruktur
```
custom-ai/
├── app.py                      # Gradio Entry Point
├── requirements.txt            # Python Dependencies
├── LICENSE.txt                 # Proprietary License
│
├── core/                       # Business Logic
│   ├── __init__.py
│   ├── project_manager.py     # Projekt-Verwaltung
│   ├── hardware_detector.py   # GPU/RAM/VRAM Detection + Warnings
│   ├── rag_processor.py       # RAG-Daten: Upload, Chunking, SQLite
│   ├── ollama_client.py       # Ollama API Integration
│   ├── sft_generator.py       # SFT-Daten via OpenRouter/Ollama
│   ├── trainer.py             # Unsloth Training + Checkpoints
│   └── exporter.py            # GGUF/Safetensors/Ollama Export
│
├── ui/                         # Gradio UI Components
│   ├── __init__.py
│   ├── tab_rag.py             # Tab 1: RAG-Datenvorbereitung
│   ├── tab_model.py           # Tab 2: Modell-Auswahl
│   ├── tab_sft.py             # Tab 3: SFT-Datenerstellung
│   ├── tab_training.py        # Tab 4: Training & Export
│   └── settings.py            # Settings-Menü
│
├── utils/                      # Helpers
│   ├── __init__.py
│   ├── config.py              # Config-Management (JSON)
│   ├── logger.py              # Logging-Utilities
│   └── validators.py          # Input-Validierung
│
├── projects/                   # User-Projekte (zur Laufzeit erstellt)
│   └── example_project/
│       ├── config.json        # Projekt-Konfiguration
│       ├── data/
│       │   ├── rag_export.db  # SQLite RAG-Datenbank
│       │   └── sft_data.jsonl # SFT-Trainingsdaten
│       ├── models/
│       │   ├── checkpoints/   # Training-Checkpoints
│       │   └── final/         # Finales Modell (GGUF/Safetensors)
│       └── logs/
│           └── training.log   # Training-Logs
│
├── docs/                       # Dokumentation
│   ├── c-ai_prd.md            # Original PRD
│   ├── implementation_plan.md # Dieser Plan
│   ├── setup_guide.md         # Installation (alle OS)
│   ├── user_manual.md         # Bedienungsanleitung
│   └── troubleshooting.md     # Häufige Probleme
│
└── tests/                      # Unit Tests (optional)
    ├── test_rag.py
    ├── test_trainer.py
    └── ...
```

### Datenfluss

```
1. Projekt anlegen → project_manager.py erstellt Ordnerstruktur

2. RAG-Vorbereitung (Tab 1):
   Docs hochladen → rag_processor.py
     → Chunking (512 Tokens, 50 Overlap)
     → SQLite speichern
     → Export-Button → Dateimanager öffnen

3. Modell-Auswahl (Tab 2):
   ollama_client.py → Liste Modelle
   User wählt Base-Modell + Ausgabeformat
   → Speichern in config.json

4. SFT-Generierung (Tab 3):
   User: API-Key + Meta-Prompts + Anzahl Samples
   → sft_generator.py → API-Calls (async)
   → Speichern: sft_data.jsonl

5. Training (Tab 4):
   trainer.py lädt:
     - Base-Modell (Ollama)
     - SFT-Daten (JSONL)
     - Hyperparameter (UI)
   → Unsloth Fine-Tuning (LoRA/QLoRA)
   → Checkpoints bei jedem Epoch
   → Export: GGUF/Safetensors/Ollama
```

---

## Implementierungs-Phasen

### Phase 1: Setup & Infrastruktur (Tag 1-2)
**Ziel**: Grundgerüst lauffähig

- [ ] Projektstruktur anlegen
- [ ] Requirements.txt mit Dependencies:
  - `gradio>=4.0.0`
  - `unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git` (inkludiert automatisch transformers, peft, datasets, trl)
  - `torch>=2.1.0` (Auto-Detection: CUDA/MPS/CPU)
  - `langchain>=0.1.0` (für Text-Chunking)
  - `langchain-text-splitters` (Chunking-Utilities)
  - `pypdf` (PDF-Parsing)
  - `python-docx` (DOCX-Parsing)
  - `aiohttp` (Async API-Calls für OpenRouter/Ollama)
  - SQLite3 (Python built-in, keine Installation nötig)
- [ ] Hardware-Detection (`hardware_detector.py`)
  - Erkennung: CUDA/MPS/CPU
  - RAM/VRAM Prüfung
  - Warnungen bei <16GB RAM oder CPU-only
- [ ] Projekt-Manager (`project_manager.py`)
  - Create/Load/Switch Project
  - Config.json Management
  - Ordnerstruktur-Generierung

**Deliverable**: `python app.py` startet leere Gradio-UI mit 4 Tabs

---

### Phase 2: Tab 1 - RAG-Datenvorbereitung (Tag 3-4)
**Ziel**: Dokumente → SQLite-Datenbank

#### UI-Komponenten (`ui/tab_rag.py`)
- File-Upload (Multi-Select: PDF, TXT, MD, DOCX)
- Tabelle: Hochgeladene Dokumente + Status
- Buttons:
  - "Verarbeiten" → Chunking + DB-Speicherung
  - "Export DB" → Dateimanager öffnen (OS-spezifisch)
- Live-Log: Verarbeitungsfortschritt

#### Backend (`core/rag_processor.py`)
- **Chunking-Logik**:
  - LangChain `RecursiveCharacterTextSplitter`
  - Chunk-Size: 512 Tokens
  - Overlap: 50 Tokens
- **SQLite-Schema**:
  ```sql
  CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```
- **Export-Funktion**: Kopiere DB nach `projects/{name}/data/rag_export.db`

**Deliverable**: Dokumente hochladen → verarbeiten → SQLite exportieren funktioniert

---

### Phase 3: Tab 2 - Modell-Auswahl (Tag 5)
**Ziel**: Ollama-Modelle listen + Konfiguration

#### UI-Komponenten (`ui/tab_model.py`)
- Dropdown: Verfügbare Ollama-Modelle (Refresh-Button)
- Radio-Buttons: Ausgabeformat
  - GGUF (für llama.cpp/Ollama)
  - Safetensors (Hugging Face Standard)
  - Ollama-Import (GGUF + automatischer `ollama create`)
- Speichern-Button → schreibt in `config.json`

#### Backend (`core/ollama_client.py`)
- **Modell-Listing**:
  ```python
  subprocess.run(["ollama", "list"], capture_output=True)
  # Parse Output → Liste von Modell-Namen
  ```
- **Validierung**: Check ob Ollama installiert/läuft
- **Config-Update**: Speichern in Projekt-Config

**Deliverable**: Ollama-Modelle werden korrekt gelistet und in Config gespeichert

---

### Phase 4: Tab 3 - SFT-Datenerstellung (Tag 6-8)
**Ziel**: Helfer-KI generiert SFT-Daten via API

#### UI-Komponenten (`ui/tab_sft.py`)
- **API-Konfiguration**:
  - Radio-Buttons: Provider-Auswahl (OpenRouter / Ollama)
  - **OpenRouter-spezifisch**:
    - Text-Input: API-Key (masked, type="password")
    - Dropdown: Modell-Auswahl (GPT-4, Claude, Llama-3, Mixtral, etc.)
    - Info-Link: "API-Key erhalten" → https://openrouter.ai/keys
  - **Ollama-spezifisch**:
    - Text-Input: Base-URL (Default: http://localhost:11434)
    - Dropdown: Lokale Modelle (von Ollama-API geladen)
    - Button: "Refresh Models"
- **Prompt-Konfiguration**:
  - Textarea: System-Prompt (anpassbar, mit Default + Reset-Button)
  - Textarea: User-Prompt-Template (anpassbar, mit Variablen wie {topic})
  - Number-Input: Anzahl Samples (1-1000, Default: 100)
  - Slider: Temperature (0.0-2.0, Default: 0.7)
- **Buttons**:
  - "Test API" → Einmalige Anfrage zur Validierung (zeigt Response)
  - "Generieren" → Batch-Generierung starten
  - "Stopp" → Generierung abbrechen (behält bereits generierte Samples)
- **Progress**:
  - Progress-Bar (0-100%)
  - Live-Log: Generierte Samples (scrollbar)
  - Counter: "42/100 Samples generiert"
- **Preview-Tabelle**: Zeige erste 10 Samples (Spalten: Instruction, Output)

#### Backend (`core/sft_generator.py`)

**OpenRouter API-Client**:
```python
import aiohttp

class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    async def generate(self, model: str, messages: list, api_key: str):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-app.com",  # Optional
            "X-Title": "Custom AI Training App"      # Optional
        }
        payload = {
            "model": model,  # z.B. "openai/gpt-4", "anthropic/claude-3"
            "messages": messages,
            "temperature": 0.7
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.BASE_URL, json=payload, headers=headers) as resp:
                return await resp.json()
```

**Verfügbare OpenRouter-Modelle** (Dropdown):
- `openai/gpt-4-turbo`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `meta-llama/llama-3-70b-instruct`
- `mistralai/mixtral-8x7b-instruct`
- `google/gemini-pro`

**Ollama API-Client**:
```python
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, model: str, prompt: str):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                return await resp.json()

    async def list_models(self):
        url = f"{self.base_url}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                return [m["name"] for m in data.get("models", [])]
```

**Async-Generierung**:
- `asyncio.Semaphore(5)` für max 5 parallele Anfragen
- Retry-Logik: 3 Versuche mit Exponential Backoff bei Fehlern
- Error-Handling: API-Limits, Timeouts, Invalid JSON

**Output-Format** (JSONL):
```json
{"instruction": "Was ist Photosynthese?", "output": "Photosynthese ist der Prozess..."}
{"instruction": "Erkläre Quantenverschränkung", "output": "Quantenverschränkung beschreibt..."}
```

**Speichern**: `projects/{name}/data/sft_data.jsonl` (append-mode für Resume)

#### Default Meta-Prompts (anpassbar)
```
System-Prompt:
"Du bist ein Experte für [THEMA]. Generiere realistische Frage-Antwort-Paare
für das Training eines Language Models. Antworte nur mit JSON."

User-Prompt:
"Generiere 1 Frage-Antwort-Paar im Format:
{\"instruction\": \"Frage\", \"output\": \"Antwort\"}"
```

**Deliverable**: API-Calls funktionieren, JSONL-Datei wird korrekt erstellt

---

### Phase 5: Tab 4 - Training & Export (Tag 9-14)
**Ziel**: Unsloth-Training mit Checkpoints + Export

#### UI-Komponenten (`ui/tab_training.py`)

**Training-Konfiguration**:
- Slider: Learning Rate (1e-5 bis 1e-3, Default: 2e-4)
- Slider: Batch Size (1, 2, 4, 8, 16 - abhängig von Hardware)
- Slider: Epochs (1-20, Default: 3)
- Slider: LoRA Rank (8, 16, 32, 64, Default: 16)
- Slider: LoRA Alpha (16, 32, 64, Default: 32)
- Checkbox: Gradient Checkpointing (für Speicher-Einsparung)
- Checkbox: 4-bit Quantisierung (QLoRA)

**Buttons**:
- "Training starten" → Disabled während Training
- "Pause" → Speichert Checkpoint, pausiert Training
- "Resume" → Lädt letzten Checkpoint
- "Abbrechen" → Stoppt Training, behält Checkpoints

**Live-Feedback**:
- Progress-Bar: Epoch-Fortschritt
- Textbox (scrollbar): Live-Logs (Loss, Lernrate, ETA)
- Metriken-Display:
  - Current Loss
  - Best Loss
  - Elapsed Time
  - ETA

**Export-Sektion** (nach Training):
- Radio: Ausgabeformat (aus Tab 2 vorausgefüllt)
- Button: "Export" → Führt Export durch
- Status: "Export erfolgreich: /path/to/model"

#### Backend (`core/trainer.py`)

**Unsloth-Setup**:
```python
from unsloth import FastLanguageModel

# Modell laden (4-bit wenn gewählt)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=config["use_4bit"]
)

# LoRA konfigurieren
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing=config["gradient_checkpointing"]
)
```

**Training-Loop**:
- `SFTTrainer` von TRL (optimiert für Supervised Fine-Tuning)
- Checkpoint-Speicherung: Jeder Epoch + bei bester Loss
- Callbacks für Live-Logging (an Gradio streamen)
- Fehlerbehandlung:
  - CUDA OOM → Warnung, Vorschlag kleinere Batch-Size
  - Keyboard-Interrupt → Checkpoint speichern

**Checkpoint-System**:
```
projects/{name}/models/checkpoints/
├── epoch_1/
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── training_args.bin
├── epoch_2/
└── best/  # Checkpoint mit niedrigster Loss
```

**Resume-Logik**:
- Prüfe ob Checkpoints existieren
- Lade letzten/besten Checkpoint
- Setze Training fort ab letztem Epoch

#### Backend (`core/exporter.py`)

**GGUF-Export**:
```python
model.save_pretrained_gguf(
    output_path,
    tokenizer,
    quantization_method="q4_k_m"  # Gängigste Quantisierung
)
```

**Safetensors-Export**:
```python
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

**Ollama-Import**:
```python
# 1. Export als GGUF
gguf_path = export_gguf(...)

# 2. Modelfile erstellen
modelfile = f"""
FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
with open("Modelfile", "w") as f:
    f.write(modelfile)

# 3. Ollama create
subprocess.run([
    "ollama", "create",
    config["custom_model_name"],
    "-f", "Modelfile"
])
```

**Deliverable**: Training funktioniert end-to-end, Checkpoints, Export in allen Formaten

---

### Phase 6: Settings & Polish (Tag 15-16)
**Ziel**: Konfigurierbarkeit + UX-Verbesserungen

#### Settings-Menü (`ui/settings.py`)
Eigener Tab oder Sidebar:

- **API-Konfiguration**:
  - OpenRouter API-Key (persistent speichern)
  - Ollama Base-URL
- **Meta-Prompt-Editor**:
  - Textarea: System-Prompt (mit Reset-to-Default)
  - Textarea: User-Prompt-Template
- **Standard-Hyperparameter**:
  - Default Learning Rate, Epochs, LoRA-Rank, etc.
- **Hardware-Override**:
  - Force CPU (für Testing)
  - Max RAM Usage (MB)
- **Speichern**: Global in `settings.json` (nicht pro Projekt)

#### UX-Verbesserungen:
- **Tooltips**: Hover-Erklärungen für Fachbegriffe (LoRA, Quantisierung, etc.)
- **Validierung**: Input-Checks vor Training-Start
  - SFT-Daten vorhanden?
  - Modell ausgewählt?
  - Genug Speicherplatz?
- **Error-Modals**: Freundliche Fehlermeldungen statt Crashes
- **Dark Mode**: Gradio-Theme anpassen (optional)

**Deliverable**: Settings funktionieren, App ist intuitiv bedienbar

---

### Phase 7: Testing & Bugfixes (Tag 17-18)
**Ziel**: Stabilität sicherstellen

- [ ] **Manuelle Tests**:
  - Kompletter Workflow: RAG → SFT → Training → Export
  - Test auf macOS (MPS)
  - Test auf Windows (CUDA - falls Zugang)
  - CPU-Fallback testen
- [ ] **Edge-Cases**:
  - Leere Inputs
  - Sehr große Dateien (>100MB)
  - Training-Abbruch mitten in Epoch
  - Ollama nicht installiert/läuft nicht
- [ ] **Performance**:
  - Training-Speed messen (Baseline dokumentieren)
  - RAM/VRAM-Usage monitoren

**Deliverable**: Stabile Version ohne kritische Bugs

---

### Phase 8: Dokumentation (Tag 19-20)
**Ziel**: User können App selbstständig nutzen

#### Setup-Guide (`docs/setup_guide.md`)
- **Prerequisites**:
  - Python 3.10+ Installation
  - Ollama Installation + erste Schritte
  - CUDA-Setup (Windows) - optional
- **Installation**:
  ```bash
  git clone <private-repo>
  cd custom-ai
  pip install -r requirements.txt
  python app.py
  ```
- **Troubleshooting**: Häufige Installations-Probleme

#### User Manual (`docs/user_manual.md`)
- **Schritt-für-Schritt-Tutorial**:
  1. Projekt anlegen
  2. RAG-Daten vorbereiten
  3. Modell wählen
  4. SFT-Daten generieren
  5. Training durchführen
  6. Modell exportieren/testen
- **Screenshots**: Von jedem Tab
- **Best Practices**:
  - Empfohlene Chunk-Sizes
  - Hyperparameter-Tuning-Tipps
  - Hardware-Anforderungen pro Modellgröße

#### Troubleshooting-Guide (`docs/troubleshooting.md`)
- **CUDA Errors**
- **OOM (Out of Memory)**
- **Langsames Training**
- **Ollama-Connection Issues**

**Deliverable**: Komplette Dokumentation für Early Access User

---

## Phase 9: Distribution Vorbereitung (SPÄTER)

### PyInstaller-Build
```bash
pyinstaller --onefile \
  --windowed \
  --name="CustomAI" \
  --icon=assets/icon.ico \
  --add-data="ui:ui" \
  app.py
```

**Challenges**:
- Unsloth Dependencies (C++ Extensions)
- PyTorch (große Binaries, 2-3GB)
- CUDA-Libraries einbetten

**Testing**: Auf frischem System ohne Python

### Docker-Setup
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "app.py"]
```

**Docker Compose**:
```yaml
services:
  custom-ai:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./projects:/app/projects
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### License System (optional)
- Hardware-ID basierte Aktivierung
- Online-Validierung (License-Server)
- Oder: Offline-Lizenzschlüssel (simpler)

---

## Hardware-Anforderungen (für Docs)

### Minimum (CPU-only):
- CPU: 4 Kerne, 2.5GHz+
- RAM: 16GB
- Speicher: 20GB frei
- **Performance**: ~50-100x langsamer als GPU

### Empfohlen (GPU):
- GPU: NVIDIA RTX 3060 (12GB VRAM) oder Apple M1/M2 (16GB unified)
- RAM: 16GB
- Speicher: 50GB frei
- **Performance**: 7B-Modell Fine-Tuning in 1-3h

### Optimal (High-End):
- GPU: NVIDIA RTX 4090 (24GB) oder Apple M2 Ultra (64GB+)
- RAM: 32GB+
- Speicher: 100GB+ SSD
- **Performance**: 13B-Modell Fine-Tuning in 2-4h

---

## Offene Fragen / Decisions Needed

1. **Lizenz-System**: Wann und wie implementieren? (Phase 9 oder später?)
2. **Cloud-Backend**: Optional Upload-to-Cloud für Heavy-Training? (Out-of-scope?)
3. **Modell-Hub**: Eigener Hub für fertige Custom-Modelle? (Later)
4. **Multi-User**: Projekt-Sharing/Collaboration? (Later)
5. **Telemetrie**: Usage-Analytics sammeln? (Privacy-Considerations)

---

## Zeitplan (Schätzung)

- **Phase 1-3**: ~1 Woche (Setup + RAG + Modell-Auswahl)
- **Phase 4-5**: ~2 Wochen (SFT-Generierung + Training - komplex!)
- **Phase 6-8**: ~1 Woche (Settings + Testing + Docs)

**Total MVP**: ~4 Wochen Solo-Entwicklung

**Distribution-Ready** (Phase 9): +1-2 Wochen

---

## Success-Kriterien

**MVP ist erfolgreich wenn**:
- [ ] User kann Projekt anlegen
- [ ] RAG-DB wird korrekt generiert und exportiert
- [ ] SFT-Daten werden via API generiert (min. 100 Samples)
- [ ] Training funktioniert auf macOS (MPS) und Windows (CUDA)
- [ ] Export in GGUF + Safetensors funktioniert
- [ ] Ollama-Import funktioniert
- [ ] Checkpoints ermöglichen Resume nach Abbruch
- [ ] Dokumentation erlaubt selbstständige Nutzung

**Distribution-Ready wenn**:
- [ ] PyInstaller-Build funktioniert auf macOS + Windows
- [ ] Docker-Image läuft mit GPU-Support
- [ ] License-System (Basic) implementiert
- [ ] Onboarding-Guide für Non-Techies

---

## Nächste Schritte

1. ✅ **Setup Git Repo** (private)
2. ⏳ **Phase 1 starten**: Projektstruktur + Hardware-Detection + Projekt-Manager
3. ⏳ **Gradio Basis-UI**: 4 leere Tabs + Projekt-Dropdown

**Ready to start coding?** 🚀
