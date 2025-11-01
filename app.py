"""
Custom AI Training App - Main Entry Point
Gradio-based UI for training small language models with RAG data.
"""

import gradio as gr
from core.hardware_detector import HardwareDetector
from core.project_manager import ProjectManager
from ui.tab_project import create_project_tab
from ui.tab_rag import create_rag_tab
from ui.tab_cpt import create_cpt_tab
from ui.tab_sft import create_sft_tab
from ui.tab_training import create_training_tab
from ui.settings import create_settings_tab

# Initialize managers
project_manager = ProjectManager()
hardware_info = None


def detect_hardware():
    """Detect hardware and return summary."""
    global hardware_info
    hardware_info = HardwareDetector.detect()
    return hardware_info.get_summary()




# Build Gradio UI
with gr.Blocks(title="Custom AI Training") as app:
    gr.Markdown("""
    # Custom AI Training App

    **4-Phase Training Pipeline:** RAG → CPT → SFT Data → SFT Training

    Train small language models with custom domain knowledge for educational purposes.
    """)

    # Hardware Info Section
    with gr.Accordion("Hardware Information", open=False):
        hardware_output = gr.Textbox(
            label="Detected Hardware",
            lines=12,
            interactive=False,
            value=detect_hardware()
        )
        gr.Button("Refresh Hardware Info").click(
            fn=detect_hardware,
            outputs=hardware_output
        )

    gr.Markdown("---")

    # Main Tabs - Project Management + 4-Phase Training Pipeline
    with gr.Tabs():
        # Tab 0: Project Management
        with gr.Tab("Project"):
            project_components = create_project_tab(project_manager)

        # Tab 1: RAG Data Preparation (wissensbasis.sqlite)
        with gr.Tab("1. RAG Data"):
            rag_components = create_rag_tab(project_manager)

        # Tab 2: CPT Training (Continued Pre-Training)
        with gr.Tab("2. CPT Training"):
            cpt_components = create_cpt_tab(project_manager)

        # Tab 3: SFT Data Generation (Q&A Pairs)
        with gr.Tab("3. SFT Data"):
            sft_components = create_sft_tab(project_manager)

        # Tab 4: SFT Training (Fine-Tuning)
        with gr.Tab("4. SFT Training"):
            training_components = create_training_tab(project_manager)

        # Tab 5: Settings
        with gr.Tab("Settings"):
            settings_components = create_settings_tab()

    gr.Markdown("""
    ---

    **Custom AI Training App** | 4-Phase Training Pipeline

    **Phase 1:** RAG Data - Upload documents → wissensbasis.sqlite
    **Phase 2:** CPT Training - Train model on RAG chunks (domain knowledge)
    **Phase 3:** SFT Data - Generate Q&A pairs from RAG data
    **Phase 4:** SFT Training - Fine-tune CPT model (response behavior)

    Built with Gradio, HuggingFace Transformers, PEFT
    """)


if __name__ == "__main__":
    print(" Starting Custom AI Training App...")
    print("\n" + "="*60)

    # Show hardware info on startup
    hw_info = HardwareDetector.detect()
    print(hw_info.get_summary())

    print("="*60)
    print("\n Launching Gradio UI...\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
