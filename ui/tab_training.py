"""
Training Tab UI
Handles model training and export.
"""

import gradio as gr
import asyncio
from pathlib import Path
from typing import Tuple
from core.project_manager import ProjectManager
from core.trainer import ModelTrainer
from core.exporter import ModelExporter
from utils.config import get_global_config


def create_training_tab(project_manager: ProjectManager):
    """
    Create training tab for model fine-tuning and export.

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    trainer_instance = None
    exporter_instance = None
    config = get_global_config()

    def get_trainer() -> Tuple[bool, ModelTrainer | None, str]:
        """Get or create trainer instance."""
        project = project_manager.get_current_project()
        if not project:
            return False, None, "No project selected"

        trainer = ModelTrainer(project.project_path)
        return True, trainer, ""

    def get_exporter() -> Tuple[bool, ModelExporter | None, str]:
        """Get or create exporter instance."""
        project = project_manager.get_current_project()
        if not project:
            return False, None, "No project selected"

        exporter = ModelExporter(project.project_path)
        return True, exporter, ""

    def check_prerequisites() -> str:
        """Check if all prerequisites are met for SFT training."""
        project = project_manager.get_current_project()

        if not project:
            return "Error: No project selected. Please create or select a project first."

        issues = []

        # Check CPT model from Tab 2
        cpt_model_path = project.cpt_model_path
        if not cpt_model_path.exists() or not list(cpt_model_path.glob("*.safetensors")):
            issues.append("CPT model not found (Tab 2) - Must complete CPT training first")

        # Check SFT data from Tab 3
        sft_data_path = project.get_sft_data_path()
        if not sft_data_path.exists():
            issues.append("SFT training data not generated (Tab 3)")
        else:
            # Count samples
            try:
                with open(sft_data_path, 'r') as f:
                    num_samples = sum(1 for line in f if line.strip())
                if num_samples == 0:
                    issues.append("SFT data file is empty")
            except Exception as e:
                issues.append(f"Error reading SFT data: {str(e)}")
                num_samples = 0

        if issues:
            return "Prerequisites not met:\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            return f"""Ready for SFT Training!

CPT Model: Available (from Tab 2)
SFT Q&A Pairs: {num_samples} (from Tab 3)

This training will teach the CPT model HOW to answer questions."""

    def load_model_for_training(use_4bit: bool, gradient_checkpointing: bool, lora_rank: int) -> str:
        """Load CPT model from Tab 2 and prepare for SFT training."""
        try:
            success, trainer, msg = get_trainer()
            if not success:
                return f"Error: {msg}"

            project = project_manager.get_current_project()

            # Load CPT model from Tab 2 (not base model!)
            cpt_model_path = project.cpt_model_path

            if not cpt_model_path.exists():
                return "Error: CPT model not found. Please complete CPT training in Tab 2 first."

            # Load CPT model (already has domain knowledge)
            success, msg = trainer.load_model(
                model_id=str(cpt_model_path),  # Load from local path, not HuggingFace
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=use_4bit
            )

            if not success:
                return f"Error loading CPT model: {msg}"

            # Add LoRA adapters for SFT (different from CPT LoRA)
            success, msg = trainer.prepare_model_for_training(
                lora_rank=int(lora_rank),
                lora_alpha=int(lora_rank),
                lora_dropout=0.0,
                use_gradient_checkpointing=gradient_checkpointing
            )

            if not success:
                return f"Error preparing for SFT: {msg}"

            return f"""CPT Model loaded successfully!

{msg}

Model has domain knowledge from CPT.
Ready for SFT training to learn response behavior."""

        except Exception as e:
            return f"Error: Failed to load CPT model: {str(e)}"

    def start_training(
        learning_rate: float,
        batch_size: int,
        epochs: int,
        lora_rank: int,
        use_4bit: bool,
        gradient_checkpointing: bool,
        progress=gr.Progress()
    ) -> str:
        """Start training process."""
        try:
            # Get trainer
            success, trainer, msg = get_trainer()
            if not success:
                return f"L {msg}"

            project = project_manager.get_current_project()
            base_model = project.config.get("base_model")

            # Load model if not already loaded
            if trainer.model is None:
                progress(0, desc="Loading model...")
                load_msg = load_model_for_training(use_4bit, gradient_checkpointing, lora_rank)
                if "L" in load_msg:
                    return load_msg

            # Get paths
            sft_data_path = project.get_sft_data_path()
            checkpoints_path = project.project_path / "models" / "checkpoints"

            # Start training
            progress(0.1, desc="Starting training...")

            success, msg = trainer.train(
                sft_data_path=sft_data_path,
                output_dir=checkpoints_path,
                learning_rate=learning_rate,
                num_train_epochs=int(epochs),
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=-1,
                logging_steps=10,
                save_steps=100,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                fp16=not use_4bit,
                bf16=False
            )

            if success:
                progress(1.0, desc="Training complete!")

                # Save final model
                final_model_path = project.project_path / "models" / "final"
                save_success, save_msg = trainer.save_final_model(final_model_path)

                return f"{msg}\n\n{save_msg}\n\nYou can now export the model below."
            else:
                return msg

        except Exception as e:
            return f"L Training failed: {str(e)}"

    def export_model(export_format: str, model_name: str, quantization: str) -> str:
        """Export trained model."""
        try:
            success, exporter, msg = get_exporter()
            if not success:
                return f"L {msg}"

            project = project_manager.get_current_project()
            final_model_path = project.project_path / "models" / "final"

            if not final_model_path.exists():
                return "L No trained model found. Please train a model first."

            if export_format == "GGUF":
                if not model_name:
                    model_name = f"{project.name}_model"

                success, result = exporter.export_to_gguf(
                    model_path=final_model_path,
                    output_name=model_name,
                    quantization=quantization
                )

                if success:
                    return f" Model exported to GGUF!\n\nFile: {result}\n\nYou can use this with llama.cpp or Ollama."
                else:
                    return result

            elif export_format == "Safetensors":
                success, result = exporter.export_to_safetensors(
                    model_path=final_model_path
                )

                if success:
                    return f" Model exported to Safetensors!\n\nDirectory: {result}\n\nYou can upload this to Hugging Face or use with transformers."
                else:
                    return result

            elif export_format == "Ollama Import":
                if not model_name:
                    model_name = f"{project.name}:latest"

                success, result = exporter.export_to_ollama(
                    model_path=final_model_path,
                    model_name=model_name,
                    quantization=quantization
                )

                if success:
                    return f"{result}\n\nRun: ollama run {model_name}"
                else:
                    return result

            else:
                return "L Unknown export format"

        except Exception as e:
            return f"L Export failed: {str(e)}"

    def get_training_status() -> str:
        """Get current training status."""
        project = project_manager.get_current_project()

        if not project:
            return "No project selected"

        final_model_path = project.project_path / "models" / "final"
        checkpoints_path = project.project_path / "models" / "checkpoints"

        status_lines = []

        # Check for final model
        if final_model_path.exists() and list(final_model_path.glob("*.safetensors")):
            status_lines.append(" Trained model available")
        else:
            status_lines.append("L No trained model")

        # Count checkpoints
        if checkpoints_path.exists():
            checkpoints = [d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            status_lines.append(f" Checkpoints: {len(checkpoints)}")
        else:
            status_lines.append(" Checkpoints: 0")

        # Check exports
        exports_path = project.project_path / "exports"
        if exports_path.exists():
            gguf_files = list(exports_path.glob("*.gguf"))
            status_lines.append(f" Exports: {len(gguf_files)} GGUF files")

        return "\n".join(status_lines)

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## Phase 4: SFT Training (Supervised Fine-Tuning)

        Fine-tune the CPT model from Tab 2 using Q&A pairs from Tab 3.

        **What happens here:**
        1. Load CPT-trained model from Tab 2
        2. Load SFT Q&A data (JSONL) from Tab 3
        3. Fine-tune with LoRA to teach the model HOW to respond
        4. Export final model for inference
        """)

        # Prerequisites check
        with gr.Accordion("Prerequisites Check", open=True):
            check_btn = gr.Button("Check Prerequisites", variant="secondary")
            prerequisites_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=4,
                value=check_prerequisites()
            )

        gr.Markdown("---")

        # Training parameters
        training_defaults = config.get_training_defaults()

        with gr.Accordion("SFT Training Parameters", open=True):
            gr.Markdown("""
            **SFT-specific recommendations:**
            - Lower learning rate than CPT (2e-4 to 3e-4)
            - Fewer epochs (1-3) to avoid overfitting
            - Smaller LoRA rank (8-16) for response behavior
            """)
            with gr.Row():
                learning_rate = gr.Slider(
                    label="Learning Rate",
                    minimum=1e-5,
                    maximum=1e-3,
                    value=training_defaults["learning_rate"],
                    step=1e-5,
                    info="Lower for stability, higher for faster learning (set defaults in Settings tab)"
                )
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=1,
                    maximum=16,
                    value=training_defaults["batch_size"],
                    step=1,
                    info="Increase if you have more GPU memory"
                )

            with gr.Row():
                epochs = gr.Slider(
                    label="Epochs",
                    minimum=1,
                    maximum=20,
                    value=training_defaults["epochs"],
                    step=1,
                    info="Number of times to iterate over the dataset"
                )
                lora_rank = gr.Slider(
                    label="LoRA Rank",
                    minimum=8,
                    maximum=64,
                    value=training_defaults["lora_rank"],
                    step=8,
                    info="Higher rank = more capacity but slower"
                )

            with gr.Row():
                use_4bit = gr.Checkbox(
                    label="4-bit Quantization (QLoRA)",
                    value=training_defaults["use_4bit"],
                    info="Use 4-bit quantization to reduce memory usage"
                )
                gradient_checkpointing = gr.Checkbox(
                    label="Gradient Checkpointing",
                    value=training_defaults["gradient_checkpointing"],
                    info="Reduces memory usage at the cost of speed"
                )

        # Training controls
        gr.Markdown("---")
        gr.Markdown("### Training Controls")

        with gr.Row():
            start_training_btn = gr.Button("Start Training", variant="primary", scale=2, size="lg")
            stop_training_btn = gr.Button("Stop Training", variant="stop", scale=1)

        training_output = gr.Textbox(
            label="Training Log",
            lines=10,
            interactive=False,
            placeholder="Click 'Start Training' to begin..."
        )

        # Export section
        gr.Markdown("---")
        with gr.Accordion("Export Model", open=False):
            gr.Markdown("""
            Export your trained model to use it with other tools.

            **Important**: You must train a model first before exporting.
            """)

            export_defaults = config.get_export_defaults()

            with gr.Row():
                export_format = gr.Radio(
                    label="Export Format",
                    choices=["GGUF", "Safetensors", "Ollama Import"],
                    value=export_defaults["format"],
                    info="Choose how to export your model (set defaults in Settings tab)"
                )

                quantization_method = gr.Dropdown(
                    label="Quantization (GGUF/Ollama only)",
                    choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                    value=export_defaults["quantization"],
                    info="Q4_K_M recommended for best balance"
                )

            model_name_input = gr.Textbox(
                label="Model Name",
                placeholder="my-tutor (for GGUF) or my-tutor:latest (for Ollama)",
                info="Leave empty for auto-generated name"
            )

            export_btn = gr.Button("Export Model", variant="secondary", size="lg")
            export_output = gr.Textbox(label="Export Status", interactive=False, lines=4)

        # Status section
        with gr.Accordion("Training Status", open=False):
            training_status_display = gr.Textbox(
                label="Current Status",
                interactive=False,
                lines=4,
                value=get_training_status()
            )
            refresh_status_btn = gr.Button("Refresh Status")

        # Wire up events
        check_btn.click(
            fn=check_prerequisites,
            outputs=[prerequisites_output]
        )

        start_training_btn.click(
            fn=start_training,
            inputs=[learning_rate, batch_size, epochs, lora_rank, use_4bit, gradient_checkpointing],
            outputs=[training_output]
        )

        export_btn.click(
            fn=export_model,
            inputs=[export_format, model_name_input, quantization_method],
            outputs=[export_output]
        )

        refresh_status_btn.click(
            fn=get_training_status,
            outputs=[training_status_display]
        )

    return {
        "training_output": training_output,
        "export_output": export_output,
        "training_status_display": training_status_display
    }
