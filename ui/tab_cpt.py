"""
CPT Training Tab UI (Continued Pre-Training)
Phase 2: Train model on RAG knowledge base to absorb domain knowledge.
"""

from __future__ import annotations

import gradio as gr
from pathlib import Path
from typing import Tuple
from core.project_manager import ProjectManager
from core.trainer import ModelTrainer
from utils.config import get_global_config


def create_cpt_tab(project_manager: ProjectManager):
    """
    Create CPT (Continued Pre-Training) tab.

    This is Phase 2 of the 4-phase training pipeline:
    1. RAG Data (Tab 1) - Upload documents, create wissensbasis.sqlite
    2. CPT Training (Tab 2 - THIS TAB) - Train base model on RAG chunks
    3. SFT Data (Tab 3) - Generate Q&A pairs from same RAG data
    4. SFT Training (Tab 4) - Fine-tune CPT model on Q&A pairs

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    config = get_global_config()
    trainer_instance = None

    def get_trainer() -> Tuple[bool, ModelTrainer | None, str]:
        """Get or create trainer instance."""
        project = project_manager.get_current_project()
        if not project:
            return False, None, "No project selected"

        trainer = ModelTrainer(project.project_path)
        return True, trainer, ""

    def check_prerequisites() -> str:
        """Check if RAG data exists for CPT training."""
        project = project_manager.get_current_project()

        if not project:
            return "Error: No project selected. Please create or select a project first."

        # Check if RAG database exists
        rag_db_path = project.project_path / "data" / "rag_db.sqlite"

        if not rag_db_path.exists():
            return """Prerequisites not met:
- RAG database not found
- Please go to Tab 1 and process documents first

The CPT training requires RAG chunks from wissensbasis.sqlite"""

        # Count chunks in database
        try:
            import sqlite3
            conn = sqlite3.connect(rag_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_chunks")
            num_chunks = cursor.fetchone()[0]
            conn.close()

            if num_chunks == 0:
                return """Prerequisites not met:
- RAG database is empty
- Please go to Tab 1 and process documents first"""

            return f"""Ready for CPT Training!

RAG Database: wissensbasis.sqlite
Total Chunks: {num_chunks}
Status: Ready

Next: Select a base model and start CPT training"""

        except Exception as e:
            return f"Error checking RAG database: {str(e)}"

    def verify_model_path(model_path: str) -> str:
        """Verify if model path exists and is valid."""
        if not model_path or not model_path.strip():
            return "Error: Please enter a model path"

        model_path = model_path.strip()

        # Check if it's a local path
        from pathlib import Path
        path = Path(model_path)

        if path.exists():
            # Check for model files
            has_config = (path / "config.json").exists()
            has_safetensors = any(path.glob("*.safetensors"))
            has_bin = any(path.glob("*.bin"))

            if has_config and (has_safetensors or has_bin):
                return f"Model verified!\n\nPath: {model_path}\nType: Local model"
            else:
                return f"Warning: Path exists but no model files found\n\nPath: {model_path}"
        else:
            # Assume it's a HuggingFace model ID
            return f"Path not found locally\n\nAssuming HuggingFace model ID: {model_path}\n\nWill attempt to download when training starts"

    def handle_model_browser(file_path) -> str:
        """Handle file browser selection and extract directory path."""
        if file_path is None:
            return ""

        # If a file was selected, get its parent directory
        # (since we want the model folder, not a specific file)
        selected_path = Path(file_path)

        if selected_path.is_file():
            # User selected a file (like config.json), use parent directory
            model_dir = selected_path.parent
        else:
            model_dir = selected_path

        return str(model_dir)

    def show_model_browser() -> dict:
        """Show the file browser component."""
        return gr.update(visible=True)

    def load_rag_chunks_for_cpt() -> Tuple[bool, Path | None, str]:
        """Load RAG chunks and prepare for CPT training."""
        project = project_manager.get_current_project()

        if not project:
            return False, None, "No project selected"

        rag_db_path = project.project_path / "data" / "rag_db.sqlite"

        if not rag_db_path.exists():
            return False, None, "RAG database not found. Process documents in Tab 1 first."

        try:
            import sqlite3
            import json

            # Connect to RAG database
            conn = sqlite3.connect(rag_db_path)
            cursor = conn.cursor()

            # Get all chunks
            cursor.execute("SELECT content FROM rag_chunks")
            chunks = cursor.fetchall()
            conn.close()

            if len(chunks) == 0:
                return False, None, "RAG database is empty"

            # Convert to training format (raw text for CPT)
            cpt_data_path = project.project_path / "data" / "cpt_data.txt"

            with open(cpt_data_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    # Each chunk on its own line for language modeling
                    f.write(chunk[0] + "\n\n")

            return True, cpt_data_path, f"Loaded {len(chunks)} chunks for CPT training"

        except Exception as e:
            return False, None, f"Failed to load RAG chunks: {str(e)}"

    def start_cpt_training(
        model_path: str,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        max_seq_length: int,
        use_4bit: bool,
        gradient_checkpointing: bool,
        progress=gr.Progress()
    ) -> str:
        """Start CPT training process."""
        try:
            # Validate model path
            if not model_path or not model_path.strip():
                return "Error: Please enter a model path"

            model_id = model_path.strip()

            project = project_manager.get_current_project()
            if not project:
                return "Error: No project selected"

            # Load RAG chunks
            progress(0.1, desc="Loading RAG chunks...")
            success, cpt_data_path, msg = load_rag_chunks_for_cpt()

            if not success:
                return f"Error: {msg}"

            # Get trainer
            success, trainer, error_msg = get_trainer()
            if not success:
                return f"Error: {error_msg}"

            # Load model
            progress(0.2, desc="Loading base model...")
            success, load_msg = trainer.load_model(
                model_id=model_id.strip(),
                max_seq_length=max_seq_length,
                load_in_4bit=use_4bit
            )

            if not success:
                return f"Error loading model: {load_msg}"

            # Prepare model for CPT (no LoRA for CPT, full model training)
            # Note: CPT typically trains full model, but we can use LoRA for efficiency
            progress(0.3, desc="Preparing model for CPT...")
            success, prep_msg = trainer.prepare_model_for_training(
                lora_rank=32,  # Higher rank for CPT to capture more knowledge
                lora_alpha=64,
                lora_dropout=0.05,
                use_gradient_checkpointing=gradient_checkpointing
            )

            if not success:
                return f"Error preparing model: {prep_msg}"

            # Start CPT training
            progress(0.4, desc="Starting CPT training...")

            cpt_output_dir = project.cpt_model_path

            success, train_msg = trainer.train_cpt(
                cpt_data_path=cpt_data_path,
                output_dir=cpt_output_dir,
                learning_rate=learning_rate,
                num_train_epochs=int(epochs),
                per_device_train_batch_size=int(batch_size),
                max_seq_length=max_seq_length,
                gradient_checkpointing=gradient_checkpointing
            )

            if success:
                progress(1.0, desc="CPT training complete!")
                return f"""CPT Training Complete!

{train_msg}

Model saved to: {cpt_output_dir}

Next Steps:
1. Go to Tab 3 to generate SFT Q&A pairs
2. Then go to Tab 4 to fine-tune this CPT model with SFT data"""
            else:
                return f"CPT Training failed: {train_msg}"

        except Exception as e:
            return f"Error during CPT training: {str(e)}"

    def get_cpt_status() -> str:
        """Get current CPT training status."""
        project = project_manager.get_current_project()

        if not project:
            return "No project selected"

        cpt_model_path = project.cpt_model_path
        rag_db_path = project.get_rag_db_path()

        status_lines = []

        # Check RAG database
        if rag_db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(rag_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM rag_chunks")
                num_chunks = cursor.fetchone()[0]
                conn.close()
                status_lines.append(f"RAG Chunks: {num_chunks}")
            except:
                status_lines.append("RAG Database: Error reading")
        else:
            status_lines.append("RAG Database: Not found")

        # Check CPT model
        if cpt_model_path.exists() and list(cpt_model_path.glob("*.safetensors")):
            status_lines.append("CPT Model: Trained")
        else:
            status_lines.append("CPT Model: Not trained yet")

        return "\n".join(status_lines)

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## Phase 2: CPT Training (Continued Pre-Training)

        Train your base model on the RAG knowledge base to absorb domain-specific knowledge.

        **What is CPT?**
        - Continued Pre-Training teaches the model factual knowledge from your documents
        - Uses language modeling (next-token prediction) on raw RAG chunks
        - Model learns the content before learning how to answer questions

        **Workflow:**
        1. RAG chunks from wissensbasis.sqlite are loaded
        2. Base model is trained to predict next tokens in the chunks
        3. Output: CPT model with domain knowledge
        4. This CPT model will be used in Tab 4 for SFT training
        """)

        # Prerequisites check
        with gr.Accordion("Prerequisites Check", open=True):
            check_btn = gr.Button("Check Prerequisites", variant="secondary")
            prerequisites_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=6,
                value=check_prerequisites()
            )

        gr.Markdown("---")

        # Model selection
        gr.Markdown("### Select Base Model")

        with gr.Accordion("Model Selection", open=True):
            gr.Markdown("**Browse for a local model or enter a HuggingFace model ID**")

            with gr.Row():
                model_path_input = gr.Textbox(
                    label="Model Path or HuggingFace ID",
                    placeholder="/path/to/your/model or microsoft/Phi-3-mini-4k-instruct",
                    info="Enter the full path to a local model folder or a HuggingFace model ID",
                    lines=1,
                    scale=4
                )

                # Gradio's native file browser for directories
                model_browser = gr.File(
                    label="Browse Models",
                    file_types=None,
                    file_count="single",
                    type="filepath",
                    scale=1,
                    visible=False  # We'll use a button instead
                )

            with gr.Row():
                browse_btn = gr.Button("📁 Browse Local Models", size="sm", scale=1)
                verify_btn = gr.Button("✓ Verify Model", size="sm", variant="secondary", scale=1)

            model_status = gr.Textbox(
                label="Model Status",
                interactive=False,
                lines=3,
                placeholder="Enter a model path and click 'Verify Model'"
            )

            gr.Markdown("""
**How to use:**
- Click "📁 Browse Local Models" to select a folder containing your downloaded model
- Or manually enter a path: `/Users/you/models/Phi-3-mini-4k-instruct`
- Or use a HuggingFace model ID: `microsoft/Phi-3-mini-4k-instruct`

**Recommended Models:**
- `microsoft/Phi-3-mini-4k-instruct` (3.8B) - Excellent for educational content
- `Qwen/Qwen2-1.5B-Instruct` (1.5B) - Very fast, good for quick iterations
- `meta-llama/Llama-3.2-3B-Instruct` (3B) - Balanced performance
- `meta-llama/Llama-3.2-1B-Instruct` (1B) - Fastest option for testing
            """)

        gr.Markdown("---")

        # CPT training parameters
        training_defaults = config.get_training_defaults()

        with gr.Accordion("CPT Training Parameters", open=True):
            gr.Markdown("""
            **CPT-specific recommendations:**
            - Higher learning rate than SFT (3e-4 to 5e-4)
            - More epochs (3-5) to absorb knowledge
            - Longer sequences if possible (2048-4096)
            """)

            with gr.Row():
                learning_rate = gr.Slider(
                    label="Learning Rate",
                    minimum=1e-5,
                    maximum=1e-3,
                    value=3e-4,  # Higher for CPT
                    step=1e-5,
                    info="Higher for CPT than SFT"
                )
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=1,
                    maximum=16,
                    value=training_defaults["batch_size"],
                    step=1,
                    info="Reduce if out of memory"
                )

            with gr.Row():
                epochs = gr.Slider(
                    label="Epochs",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    info="CPT typically needs 3-5 epochs"
                )
                max_seq_length = gr.Slider(
                    label="Max Sequence Length",
                    minimum=512,
                    maximum=4096,
                    value=2048,
                    step=512,
                    info="Longer = more context"
                )

            with gr.Row():
                use_4bit = gr.Checkbox(
                    label="4-bit Quantization",
                    value=False,
                    info="Not available on MPS, uses FP16"
                )
                gradient_checkpointing = gr.Checkbox(
                    label="Gradient Checkpointing",
                    value=True,
                    info="Recommended for CPT to save memory"
                )

        # Training controls
        gr.Markdown("---")
        gr.Markdown("### Start CPT Training")

        start_cpt_btn = gr.Button("Start CPT Training", variant="primary", size="lg")

        cpt_output = gr.Textbox(
            label="CPT Training Log",
            lines=12,
            interactive=False,
            placeholder="Click 'Start CPT Training' to begin..."
        )

        # Status section
        with gr.Accordion("CPT Status", open=False):
            cpt_status_display = gr.Textbox(
                label="Current Status",
                interactive=False,
                lines=4,
                value=get_cpt_status()
            )
            refresh_status_btn = gr.Button("Refresh Status")

        # Wire up events
        check_btn.click(
            fn=check_prerequisites,
            outputs=[prerequisites_output]
        )

        # Browse button shows file browser
        browse_btn.click(
            fn=show_model_browser,
            outputs=[model_browser]
        )

        # When file is selected in browser, update path
        model_browser.change(
            fn=handle_model_browser,
            inputs=[model_browser],
            outputs=[model_path_input]
        )

        # Model verification
        verify_btn.click(
            fn=verify_model_path,
            inputs=[model_path_input],
            outputs=[model_status]
        )

        # CPT Training
        start_cpt_btn.click(
            fn=start_cpt_training,
            inputs=[
                model_path_input,
                learning_rate,
                batch_size,
                epochs,
                max_seq_length,
                use_4bit,
                gradient_checkpointing
            ],
            outputs=[cpt_output]
        )

        refresh_status_btn.click(
            fn=get_cpt_status,
            outputs=[cpt_status_display]
        )

    return {
        "model_path_input": model_path_input,
        "cpt_output": cpt_output,
        "cpt_status_display": cpt_status_display
    }
