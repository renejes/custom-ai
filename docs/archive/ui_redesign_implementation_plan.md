# UI Redesign & Feature Enhancement - Implementation Plan

## Overview

This document outlines the implementation plan for redesigning the UI and adding multi-database support, local model management, and improved chunk selection features.

**Goals:**
1. Multi-database support for flexible knowledge management
2. Local model management (no HuggingFace download dependency)
3. DataFrame-based chunk selection for better data control
4. Card-based UI layout with clear visual hierarchy
5. Numbered steps and better information architecture

---

## Phase 1: Tab 1 - Multi-Database System

### Current State
- Single database per project: `rag_db.sqlite`
- No option to name database
- No support for multiple knowledge bases
- No merge functionality

### Target State
- User can name databases (e.g., `klasse_1_2.sqlite`, `bio_grundlagen.sqlite`)
- Multiple databases per project
- Merge functionality to combine databases
- Database selection for CPT/SFT training

---

### 1.1 Database Naming

**File:** `ui/tab_rag.py`

**Changes:**
1. Add input field for database name
   - Default: `wissensbasis.sqlite`
   - Validation: Only alphanumeric + underscores
   - Auto-append `.sqlite` if missing

2. Store database name in session state
   - Pass to RAGProcessor on file processing
   - Save in project config

**UI Structure:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 1️⃣ Step 1: Database Configuration                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                          ┃
┃ Database Name: [wissensbasis] .sqlite                   ┃
┃ 💡 Tip: Use descriptive names like 'bio_klasse7'       ┃
┃                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Implementation:**
```python
def create_rag_tab(project_manager: ProjectManager):
    # Add database name input
    db_name_input = gr.Textbox(
        label="Database Name",
        value="wissensbasis",
        placeholder="e.g., bio_klasse7, mathe_grundlagen",
        info="Name for this knowledge database"
    )

    # Modify process_files function
    def process_files_with_db_name(files, chunk_size, overlap, db_name):
        if not db_name.strip():
            db_name = "wissensbasis"

        # Sanitize name
        db_name = sanitize_db_name(db_name)

        # Ensure .sqlite extension
        if not db_name.endswith('.sqlite'):
            db_name += '.sqlite'

        # Pass to RAGProcessor
        rag_processor = RAGProcessor(project_path, db_name=db_name)
        # ... rest of processing
```

---

### 1.2 Multi-Database Management

**File:** `ui/tab_rag.py`

**Changes:**
1. Add "Database Management" section at bottom of tab
2. List all `.sqlite` files in project's `data/` folder
3. Show metadata for each database:
   - Name
   - Number of chunks
   - Total size
   - Created date
   - Sources (which files)

**UI Structure:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📊 Database Management                                   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                          ┃
┃ Existing Databases:                                      ┃
┃                                                          ┃
┃ ☑ klasse_1_2.sqlite        │ 450 chunks │ 2.1 MB       ┃
┃ ☑ klasse_3_4.sqlite        │ 380 chunks │ 1.8 MB       ┃
┃ ☑ klasse_5_6.sqlite        │ 520 chunks │ 2.4 MB       ┃
┃ ☐ klasse_7_8.sqlite        │ 610 chunks │ 2.9 MB       ┃
┃ ☑ klasse_9.sqlite          │ 340 chunks │ 1.6 MB       ┃
┃                                                          ┃
┃ [🔗 Merge Selected] [🗑️ Delete] [📋 View Details]       ┃
┃                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Implementation:**
```python
def list_databases(project_manager):
    """List all databases in project."""
    project = project_manager.get_current_project()
    if not project:
        return []

    data_path = project.project_path / "data"
    databases = []

    for db_file in data_path.glob("*.sqlite"):
        # Get metadata
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rag_chunks")
        num_chunks = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT source FROM rag_chunks")
        sources = [row[0] for row in cursor.fetchall()]
        conn.close()

        databases.append({
            "name": db_file.name,
            "path": str(db_file),
            "num_chunks": num_chunks,
            "size_mb": db_file.stat().st_size / (1024 * 1024),
            "sources": sources,
            "created": datetime.fromtimestamp(db_file.stat().st_ctime)
        })

    return databases
```

---

### 1.3 Database Merge Functionality

**File:** `core/rag_processor.py` (new method)

**Functionality:**
- Merge multiple SQLite databases into one
- Preserve all metadata (source, chunk_index)
- Optional: Deduplicate identical chunks
- Show progress during merge

**Implementation:**
```python
class RAGProcessor:
    def merge_databases(
        self,
        source_db_paths: List[Path],
        output_db_name: str,
        deduplicate: bool = True
    ) -> Tuple[bool, str]:
        """
        Merge multiple databases into one.

        Args:
            source_db_paths: List of paths to source databases
            output_db_name: Name for merged database
            deduplicate: Remove duplicate chunks

        Returns:
            Tuple of (success, message)
        """
        try:
            output_path = self.project_path / "data" / output_db_name

            # Create new database
            conn_out = sqlite3.connect(output_path)
            cursor_out = conn_out.cursor()

            # Create table schema
            cursor_out.execute("""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT,
                    chunk_index INTEGER,
                    metadata TEXT,
                    original_db TEXT
                )
            """)

            total_chunks = 0
            seen_content = set() if deduplicate else None

            # Merge each database
            for source_db in source_db_paths:
                conn_in = sqlite3.connect(source_db)
                cursor_in = conn_in.cursor()

                cursor_in.execute("SELECT content, source, chunk_index, metadata FROM rag_chunks")

                for row in cursor_in.fetchall():
                    content, source, chunk_index, metadata = row

                    # Check for duplicates
                    if deduplicate:
                        if content in seen_content:
                            continue
                        seen_content.add(content)

                    # Insert into merged database
                    cursor_out.execute("""
                        INSERT INTO rag_chunks (content, source, chunk_index, metadata, original_db)
                        VALUES (?, ?, ?, ?, ?)
                    """, (content, source, chunk_index, metadata, source_db.name))

                    total_chunks += 1

                conn_in.close()

            conn_out.commit()
            conn_out.close()

            msg = f"Successfully merged {len(source_db_paths)} databases into {output_db_name}"
            msg += f"\nTotal chunks: {total_chunks}"

            if deduplicate:
                msg += "\nDuplicates removed"

            return True, msg

        except Exception as e:
            return False, f"Merge failed: {str(e)}"
```

**UI Integration:**
```python
def merge_selected_databases(selected_dbs, output_name, deduplicate):
    """Merge selected databases."""
    if not selected_dbs:
        return "Error: No databases selected"

    if not output_name.strip():
        return "Error: Please provide output database name"

    # Get full paths
    db_paths = [Path(db) for db in selected_dbs]

    # Merge
    processor = RAGProcessor(project.project_path)
    success, msg = processor.merge_databases(
        source_db_paths=db_paths,
        output_db_name=output_name,
        deduplicate=deduplicate
    )

    return msg
```

---

### 1.4 Database Selection in Tab 2 & Tab 3

**Files:** `ui/tab_cpt.py`, `ui/tab_sft.py`

**Changes:**
1. Add dropdown to select database
2. Show available databases from project
3. Default to most recent or `wissensbasis.sqlite`

**UI Structure (Tab 2 - CPT):**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 1️⃣ Step 1: Select Knowledge Base                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                          ┃
┃ Database: [▼ merged_klasse_1-9.sqlite]                  ┃
┃                                                          ┃
┃ ℹ️ 2300 chunks | 10.8 MB | 5 sources                    ┃
┃                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Implementation:**
```python
def get_available_databases(project_manager):
    """Get list of databases for dropdown."""
    project = project_manager.get_current_project()
    if not project:
        return []

    data_path = project.project_path / "data"
    return [db.name for db in data_path.glob("*.sqlite")]

# In create_cpt_tab():
database_selector = gr.Dropdown(
    label="Select Knowledge Base",
    choices=get_available_databases(project_manager),
    value="wissensbasis.sqlite",
    info="Choose which database to use for CPT training"
)
```

---

## Phase 2: Tab 2 - Local Model Management

### Current State
- Model ID input (HuggingFace)
- Automatic download on training start
- No local model support
- No token management

### Target State
- Local model folder selection
- Model info display (size, type, parameters)
- Copy to project folder with progress
- No HuggingFace dependency (MVP)

---

### 2.1 Local Model Browser

**File:** `ui/tab_cpt.py`

**Changes:**
1. Replace "Model ID input" with "Select Local Model" button
2. Add file/folder picker
3. Validate model folder (check for required files)
4. Display model information

**UI Structure:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 2️⃣ Step 2: Select Base Model                            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                          ┃
┃ [📂 Browse Local Model Folder...]                       ┃
┃                                                          ┃
┃ Selected Path:                                           ┃
┃ /Users/name/models/Phi-3-mini-4k-instruct               ┃
┃                                                          ┃
┃ ╔════════════════════════════════════════════╗          ┃
┃ ║ 📋 Model Information                      ║          ┃
┃ ║                                            ║          ┃
┃ ║ Name: Phi-3-mini-4k-instruct              ║          ┃
┃ ║ Type: Safetensors                         ║          ┃
┃ ║ Size: 7.4 GB                              ║          ┃
┃ ║ Files: model.safetensors, config.json... ║          ┃
┃ ║                                            ║          ┃
┃ ║ Status: ✅ Valid model format              ║          ┃
┃ ╚════════════════════════════════════════════╝          ┃
┃                                                          ┃
┃ [📋 Load This Model]                                    ┃
┃                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

▼ Quick Select Recommendations (Optional)
```

**Implementation:**
```python
def validate_model_folder(folder_path: str) -> Tuple[bool, Dict, str]:
    """
    Validate if folder contains valid model files.

    Args:
        folder_path: Path to model folder

    Returns:
        Tuple of (is_valid, model_info, error_message)
    """
    path = Path(folder_path)

    if not path.exists():
        return False, {}, "Folder does not exist"

    # Check for required files
    required_files = {
        "safetensors": ["model.safetensors", "model-00001-of-*.safetensors"],
        "config": ["config.json"],
        "tokenizer": ["tokenizer.json", "tokenizer_config.json"]
    }

    found_files = {}
    model_type = None

    # Check for safetensors
    if list(path.glob("*.safetensors")):
        model_type = "safetensors"
        found_files["model"] = list(path.glob("*.safetensors"))

    # Check for config
    if (path / "config.json").exists():
        found_files["config"] = path / "config.json"

    # Check for tokenizer
    tokenizer_files = list(path.glob("tokenizer*"))
    if tokenizer_files:
        found_files["tokenizer"] = tokenizer_files

    if not model_type:
        return False, {}, "No model files found (looking for .safetensors)"

    if "config" not in found_files:
        return False, {}, "config.json not found"

    if "tokenizer" not in found_files:
        return False, {}, "Tokenizer files not found"

    # Calculate size
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    # Try to get model name from config
    try:
        import json
        with open(found_files["config"], 'r') as f:
            config = json.load(f)
            model_name = config.get("_name_or_path", path.name)
    except:
        model_name = path.name

    model_info = {
        "name": model_name,
        "path": str(path),
        "type": model_type,
        "size_gb": total_size / (1024**3),
        "files": [f.name for f in path.rglob("*") if f.is_file()],
        "num_files": len(list(path.rglob("*")))
    }

    return True, model_info, ""

def load_local_model(model_path: str, project_manager) -> str:
    """
    Copy local model to project folder.

    Args:
        model_path: Source model folder
        project_manager: Project manager instance

    Returns:
        Status message
    """
    project = project_manager.get_current_project()
    if not project:
        return "Error: No project selected"

    # Validate
    is_valid, model_info, error = validate_model_folder(model_path)
    if not is_valid:
        return f"Error: {error}"

    # Destination
    dest_path = project.project_path / "models" / "base_model"
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy with progress
    import shutil
    try:
        # Copy all files
        for file in Path(model_path).rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(model_path)
                dest_file = dest_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest_file)

        return f"""Model loaded successfully!

Name: {model_info['name']}
Type: {model_info['type']}
Size: {model_info['size_gb']:.2f} GB
Files: {model_info['num_files']}

Model copied to: {dest_path}
Ready for CPT training!"""

    except Exception as e:
        return f"Error copying model: {str(e)}"
```

**Gradio Integration:**
```python
# In create_cpt_tab():
model_folder_input = gr.Textbox(
    label="Model Folder Path",
    placeholder="/path/to/model/folder",
    interactive=True
)

browse_btn = gr.Button("📂 Browse Local Model Folder")
model_info_display = gr.JSON(label="Model Information", visible=False)
load_model_btn = gr.Button("📋 Load This Model", visible=False)
load_status = gr.Textbox(label="Status", interactive=False)

# Events
def on_browse_click():
    # Open file dialog (Gradio limitation: needs external library)
    # Alternative: User pastes path manually
    return gr.update()

def on_validate_model(folder_path):
    is_valid, info, error = validate_model_folder(folder_path)

    if is_valid:
        return (
            gr.update(value=info, visible=True),
            gr.update(visible=True),
            f"✅ Valid model found: {info['name']}"
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ {error}"
        )

model_folder_input.change(
    fn=on_validate_model,
    inputs=[model_folder_input],
    outputs=[model_info_display, load_model_btn, load_status]
)

load_model_btn.click(
    fn=load_local_model,
    inputs=[model_folder_input],
    outputs=[load_status]
)
```

---

### 2.2 Model Recommendations (Optional Quick Select)

**File:** `ui/tab_cpt.py`

**Changes:**
1. Keep existing recommended models list
2. Add note: "Download these models first, then select folder"
3. Provide download links

**UI Structure:**
```
▼ 💡 Recommended Models (Download First)

┌──────────────────────────────────────────────────────┐
│ Phi-3 Mini (3.8B) - Best for education              │
│ Size: 7.4 GB | Download: huggingface.co/microsoft/..│
│ [📥 Copy HF Link]                                    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Qwen2 (1.5B) - Fast training                        │
│ Size: 3.1 GB | Download: huggingface.co/Qwen/...    │
│ [📥 Copy HF Link]                                    │
└──────────────────────────────────────────────────────┘
```

---

## Phase 3: Tab 3 - DataFrame Chunk Selection

### Current State
- Slider to browse chunks one by one
- No multi-select
- No table view
- Chunks selected automatically (evenly distributed)

### Target State
- DataFrame/Table view with all chunks
- Checkboxes for selection
- Filter by source file
- Sort by various criteria
- Bulk selection actions
- Generate Q&A only from selected chunks

---

### 3.1 DataFrame View

**File:** `ui/tab_sft.py`

**Changes:**
1. Replace slider with Gradio DataFrame
2. Add checkbox column
3. Show all relevant metadata
4. Enable sorting and filtering

**UI Structure:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 2️⃣ Step 2: Select Chunks for Q&A Generation            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                          ┃
┃ [🔄 Load Chunks from Database]                          ┃
┃                                                          ┃
┃ ╔════════════════════════════════════════════════════╗  ┃
┃ ║ Filter: [▼ All Sources] [Search: ____]            ║  ┃
┃ ║                                                    ║  ┃
┃ ║ [☑ Select All] [☐ Clear] [🎲 Random 50]          ║  ┃
┃ ╚════════════════════════════════════════════════════╝  ┃
┃                                                          ┃
┃ ┌─────────────────────────────────────────────────────┐ ┃
┃ │☑│ ID │ Source          │ Size │ Content Preview   │ ┃
┃ ├─────────────────────────────────────────────────────┤ ┃
┃ │☑│ 1  │ bio_ch1.pdf     │ 450  │ Photosynthesis... │ ┃
┃ │☑│ 2  │ bio_ch1.pdf     │ 420  │ Cellular resp...  │ ┃
┃ │☐│ 3  │ bio_ch2.pdf     │ 380  │ DNA structure...  │ ┃
┃ │☑│ 4  │ bio_ch2.pdf     │ 510  │ Protein synth...  │ ┃
┃ │☐│ 5  │ math_basics.pdf │ 290  │ Algebra rules...  │ ┃
┃ └─────────────────────────────────────────────────────┘ ┃
┃                                                          ┃
┃ Selected: 3 of 2300 chunks                              ┃
┃                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Implementation:**
```python
def load_chunks_for_dataframe(database_path: Path) -> pd.DataFrame:
    """
    Load chunks into DataFrame for table view.

    Args:
        database_path: Path to SQLite database

    Returns:
        Pandas DataFrame with chunks
    """
    import pandas as pd
    import sqlite3

    conn = sqlite3.connect(database_path)

    # Load all chunks
    df = pd.read_sql_query("""
        SELECT
            id,
            source,
            chunk_index,
            SUBSTR(content, 1, 100) as preview,
            LENGTH(content) as size,
            content
        FROM rag_chunks
        ORDER BY id
    """, conn)

    conn.close()

    # Add selection column
    df.insert(0, 'select', False)

    return df

def filter_dataframe(df: pd.DataFrame, source_filter: str, search_term: str) -> pd.DataFrame:
    """Filter DataFrame by source and search term."""
    filtered = df.copy()

    # Filter by source
    if source_filter and source_filter != "All Sources":
        filtered = filtered[filtered['source'] == source_filter]

    # Filter by search term
    if search_term:
        filtered = filtered[
            filtered['content'].str.contains(search_term, case=False, na=False) |
            filtered['preview'].str.contains(search_term, case=False, na=False)
        ]

    return filtered

# In create_sft_tab():
load_chunks_btn = gr.Button("🔄 Load Chunks from Database")

# Filter controls
with gr.Row():
    source_filter = gr.Dropdown(
        label="Filter by Source",
        choices=["All Sources"],
        value="All Sources"
    )
    search_box = gr.Textbox(
        label="Search in chunks",
        placeholder="Enter search term..."
    )

# Bulk actions
with gr.Row():
    select_all_btn = gr.Button("☑ Select All")
    clear_all_btn = gr.Button("☐ Clear All")
    random_select_btn = gr.Button("🎲 Random 50")

# DataFrame
chunks_dataframe = gr.Dataframe(
    headers=["☑", "ID", "Source", "Size", "Preview"],
    datatype=["bool", "number", "str", "number", "str"],
    interactive=True,
    wrap=True,
    label="Chunks"
)

selection_count = gr.Markdown("Selected: 0 chunks")

# Events
def on_load_chunks(database_name):
    # Load chunks
    df = load_chunks_for_dataframe(database_path)

    # Get unique sources for filter
    sources = ["All Sources"] + df['source'].unique().tolist()

    return (
        df[['select', 'id', 'source', 'size', 'preview']],
        gr.update(choices=sources)
    )

def on_select_all(df):
    df['select'] = True
    return df, f"Selected: {len(df)} chunks"

def on_clear_all(df):
    df['select'] = False
    return df, "Selected: 0 chunks"

def on_random_select(df, n=50):
    df['select'] = False
    if len(df) > n:
        indices = np.random.choice(len(df), n, replace=False)
        df.loc[indices, 'select'] = True
    else:
        df['select'] = True

    count = df['select'].sum()
    return df, f"Selected: {count} chunks"

load_chunks_btn.click(
    fn=on_load_chunks,
    inputs=[database_selector],
    outputs=[chunks_dataframe, source_filter]
)

select_all_btn.click(
    fn=on_select_all,
    inputs=[chunks_dataframe],
    outputs=[chunks_dataframe, selection_count]
)

# ... similar for other buttons
```

---

### 3.2 Generate Q&A from Selected Chunks

**File:** `ui/tab_sft.py`

**Changes:**
1. Modify `generate_qa_from_chunks()` to use only selected chunks
2. Pass DataFrame with selections
3. Filter to selected chunks only

**Implementation:**
```python
def generate_qa_from_selected_chunks(
    chunks_df: pd.DataFrame,
    provider: str,
    api_key: str,
    ollama_url: str,
    openrouter_model: str,
    ollama_model: str,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    qa_style: str,
    progress=gr.Progress()
) -> str:
    """Generate Q&A pairs from selected chunks only."""

    # Filter to selected chunks
    selected_chunks = chunks_df[chunks_df['select'] == True]

    if len(selected_chunks) == 0:
        return "Error: No chunks selected. Please select at least one chunk."

    num_samples = len(selected_chunks)

    # Load full content for selected chunks
    project = project_manager.get_current_project()
    rag_db_path = project.project_path / "data" / selected_database

    conn = sqlite3.connect(rag_db_path)
    cursor = conn.cursor()

    selected_ids = selected_chunks['id'].tolist()
    placeholders = ','.join('?' * len(selected_ids))

    cursor.execute(f"""
        SELECT id, content, source, chunk_index
        FROM rag_chunks
        WHERE id IN ({placeholders})
    """, selected_ids)

    chunks_with_content = []
    for row in cursor.fetchall():
        chunks_with_content.append({
            "id": row[0],
            "content": row[1],
            "source": row[2],
            "chunk_index": row[3]
        })

    conn.close()

    # Rest of Q&A generation logic...
    # (same as before but using chunks_with_content)

    return f"Generated {len(generated_samples)} Q&A pairs from {num_samples} selected chunks"
```

---

## Phase 4: UI Redesign - Card-Based Layout

### Current State
- Multiple Accordions and Boxes
- No clear visual hierarchy
- Hard to see workflow progression
- Inconsistent styling

### Target State
- Numbered steps (1️⃣ 2️⃣ 3️⃣ 4️⃣)
- Card-based layout with clear sections
- Color-coded by importance
- Better spacing and grouping
- Consistent design across all tabs

---

### 4.1 Design System

**File:** `app.py` (add custom theme and CSS)

**Theme Configuration:**
```python
# Custom theme
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=("Inter", "system-ui", "sans-serif"),
    spacing_size="lg",
    radius_size="md"
)

# Custom CSS
custom_css = """
/* Card styling */
.step-card {
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    background: white;
}

.step-card-header {
    font-size: 18px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Step numbers */
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #3b82f6;
    color: white;
    font-weight: 700;
    font-size: 16px;
}

/* Status indicators */
.status-success {
    color: #10b981;
    font-weight: 600;
}

.status-warning {
    color: #f59e0b;
    font-weight: 600;
}

.status-error {
    color: #ef4444;
    font-weight: 600;
}

/* Action buttons */
.primary-action {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    font-weight: 600;
    font-size: 16px;
    padding: 16px 32px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: transform 0.2s;
}

.primary-action:hover {
    transform: translateY(-2px);
}

/* Tab styling */
.tab-title {
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Better spacing */
.gradio-container {
    max-width: 1400px;
    margin: 0 auto;
}

.tab-content {
    padding: 32px 24px;
}

/* Compact accordions */
.compact-accordion .label-wrap {
    padding: 12px 16px;
}
"""

# Apply to app
with gr.Blocks(theme=custom_theme, css=custom_css, title="Custom AI Training") as app:
    # ... rest of app
```

---

### 4.2 Tab Structure Template

**Standard structure for each tab:**

```python
def create_tab_template(tab_name: str, tab_icon: str, description: str):
    """Template for consistent tab layout."""

    with gr.Column(elem_classes="tab-content"):
        # Tab header
        gr.Markdown(f"""
        # {tab_icon} {tab_name}

        {description}
        """)

        gr.Markdown("---")

        # Step 1
        with gr.Group(elem_classes="step-card"):
            gr.Markdown("""
            <div class="step-card-header">
                <span class="step-number">1</span>
                <span>First Step Title</span>
            </div>
            """, elem_classes="step-card-header")

            # Step 1 content
            # ...

        # Step 2
        with gr.Group(elem_classes="step-card"):
            gr.Markdown("""
            <div class="step-card-header">
                <span class="step-number">2</span>
                <span>Second Step Title</span>
            </div>
            """)

            # Step 2 content
            # ...

        # Action step (highlighted)
        with gr.Group(elem_classes="step-card primary-action-card"):
            gr.Markdown("""
            <div class="step-card-header">
                <span class="step-number">🚀</span>
                <span>Take Action</span>
            </div>
            """)

            # Action button
            action_btn = gr.Button("Start Process", elem_classes="primary-action")

        # Collapsible sections for advanced/optional
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            # Advanced options
            pass

        with gr.Accordion("📊 Status & Logs", open=False):
            # Status info
            pass
```

---

### 4.3 Tab-Specific Redesigns

**Tab 1 Example:**
```python
def create_rag_tab(project_manager: ProjectManager):
    with gr.Column(elem_classes="tab-content"):
        gr.Markdown("""
        # 📊 Phase 1: RAG Data Preparation

        Build your knowledge base by uploading documents and creating databases.
        """)

        gr.Markdown("---")

        # Step 1: Database Config
        with gr.Group(elem_classes="step-card"):
            gr.HTML('<div class="step-card-header"><span class="step-number">1</span> Database Configuration</div>')

            with gr.Row():
                db_name = gr.Textbox(
                    label="Database Name",
                    value="wissensbasis",
                    scale=3
                )
                gr.Markdown("`.sqlite`", scale=1)

            gr.Markdown("💡 **Tip:** Use descriptive names like `bio_klasse7` or `mathe_grundlagen`")

        # Step 2: Upload Files
        with gr.Group(elem_classes="step-card"):
            gr.HTML('<div class="step-card-header"><span class="step-number">2</span> Upload Documents</div>')

            file_upload = gr.File(
                label="Select Files",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"]
            )

            with gr.Row():
                chunk_size = gr.Slider(label="Chunk Size", minimum=100, maximum=1000, value=500)
                chunk_overlap = gr.Slider(label="Overlap", minimum=0, maximum=200, value=50)

        # Step 3: Process
        with gr.Group(elem_classes="step-card"):
            gr.HTML('<div class="step-card-header"><span class="step-number">🚀</span> Process Documents</div>')

            process_btn = gr.Button("📥 Process Files", elem_classes="primary-action", size="lg")
            process_output = gr.Textbox(label="Status", lines=6)

        # Database Management (collapsible)
        gr.Markdown("---")

        with gr.Accordion("📊 Database Management", open=False):
            gr.Markdown("### Existing Databases")

            databases_df = gr.Dataframe(
                headers=["Select", "Name", "Chunks", "Size", "Created"],
                label="Databases"
            )

            with gr.Row():
                merge_btn = gr.Button("🔗 Merge Selected")
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
                refresh_btn = gr.Button("🔄 Refresh")
```

---

## Phase 5: Implementation Order

### Week 1: Multi-Database (Tab 1)
1. **Day 1-2:** Database naming + validation
2. **Day 3-4:** Database listing + metadata
3. **Day 5:** Merge functionality
4. **Testing:** Create, merge, use in Tab 2/3

### Week 2: Local Model (Tab 2)
1. **Day 1-2:** Model folder validation + info extraction
2. **Day 3:** Model copying with progress
3. **Day 4:** UI integration + testing
4. **Day 5:** Quick select recommendations

### Week 3: DataFrame Chunks (Tab 3)
1. **Day 1-2:** DataFrame implementation + loading
2. **Day 3:** Filtering + bulk actions
3. **Day 4:** Q&A generation from selected chunks
4. **Day 5:** Testing + edge cases

### Week 4: UI Redesign (All Tabs)
1. **Day 1:** Theme + CSS setup
2. **Day 2:** Tab 1 + 2 redesign
3. **Day 3:** Tab 3 + 4 redesign
4. **Day 4:** Tab 5 (Settings) redesign
5. **Day 5:** Polish + consistency check

---

## Testing Strategy

### Unit Tests
```python
# test_database_merge.py
def test_merge_databases():
    """Test merging multiple databases."""
    pass

def test_merge_with_deduplication():
    """Test merge removes duplicates."""
    pass

# test_model_validation.py
def test_validate_valid_model():
    """Test validation accepts valid model."""
    pass

def test_validate_invalid_model():
    """Test validation rejects invalid model."""
    pass

# test_chunk_selection.py
def test_load_chunks_dataframe():
    """Test loading chunks into DataFrame."""
    pass

def test_filter_chunks_by_source():
    """Test filtering chunks."""
    pass
```

### Integration Tests
1. Full workflow: Create DB → Merge → CPT Training
2. Local model: Validate → Copy → Train
3. Chunk selection: Load → Select → Generate Q&A

### User Acceptance Tests
1. Can user create and name databases?
2. Can user merge multiple databases?
3. Can user select local model and train?
4. Can user select specific chunks for Q&A?
5. Is UI hierarchy clear?

---

## Success Metrics

### Functional
- ✅ Multi-database support working
- ✅ Merge functionality tested with 5+ databases
- ✅ Local model loading working
- ✅ Chunk selection with DataFrame working
- ✅ All existing features still functional

### UX
- ✅ Clear visual hierarchy (user testing)
- ✅ Workflow is intuitive (no documentation needed for basic flow)
- ✅ Response time < 2s for all UI interactions
- ✅ Informative error messages

### Performance
- ✅ Database merge < 10s for 10k chunks
- ✅ Model copy with progress indicator
- ✅ DataFrame loads < 3s for 5k chunks
- ✅ UI remains responsive during operations

---

## Future Enhancements (Post-MVP)

### Phase 6: HuggingFace Integration
- Token management in settings
- Direct download with progress
- Model search/browse

### Phase 7: Advanced Features
- Chunk editing in DataFrame
- Duplicate detection
- Model comparison tool
- Training metrics dashboard
- Chat interface for testing

### Phase 8: Export & Deployment
- GGUF export (with llama.cpp)
- Ollama integration
- Model card generation
- Deployment scripts

---

## Dependencies Update

**New Python packages needed:**
```txt
# Already have:
gradio>=4.0.0
sqlite3 (built-in)
pandas>=2.0.0

# May need:
tk  # For file dialog (optional)
```

**No new major dependencies required!**

---

## Risk Mitigation

### Risk 1: Gradio DataFrame Limitations
**Mitigation:** Use Gradio's native DataFrame component, fallback to simple table if issues

### Risk 2: Large Model Copy Times
**Mitigation:** Implement progress bars, allow cancellation, copy in chunks

### Risk 3: Database Merge Performance
**Mitigation:** Batch inserts (1000 chunks at a time), show progress, allow background processing

### Risk 4: File Dialog Platform Issues
**Mitigation:** Manual path input as fallback, clear instructions for users

---

## Documentation Updates Needed

1. **User Guide:**
   - How to create and merge databases
   - How to prepare local models
   - How to select chunks for Q&A

2. **Developer Guide:**
   - Database schema changes
   - Model validation logic
   - UI component structure

3. **README Updates:**
   - Multi-database workflow
   - Local model requirements
   - New UI screenshots

---

## Conclusion

This implementation plan provides a clear roadmap for:
1. ✅ Multi-database support with merge functionality
2. ✅ Local model management (no HF dependency)
3. ✅ DataFrame-based chunk selection
4. ✅ Improved card-based UI with clear hierarchy

**Total estimated time:** 4 weeks (1 developer)

**MVP features prioritized:**
- Multi-database (high priority)
- Local model (high priority)
- DataFrame chunks (high priority)
- UI redesign (medium priority - can be iterative)

**Next steps:**
1. Review and approve plan
2. Set up development environment
3. Start with Phase 1 (Multi-Database)
4. Iterate based on user feedback
