"""
Custom AI Training App - Main Entry Point
Gradio-based UI for training small language models with RAG data.
"""

import gradio as gr
from core.hardware_detector import HardwareDetector
from core.project_manager import ProjectManager
from ui.tab_rag import create_rag_tab
from ui.tab_model import create_model_tab
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


def create_new_project(name: str, description: str):
    """Create a new project."""
    if not name.strip():
        return "❌ Error: Project name cannot be empty", gr.update()

    project = project_manager.create_project(name.strip(), description.strip())

    if project:
        projects = project_manager.list_projects()
        return f"✅ Project '{name}' created successfully!", gr.update(choices=projects, value=name)
    else:
        return f"❌ Error: Project '{name}' already exists or name is invalid", gr.update()


def load_project(name: str):
    """Load an existing project."""
    if not name:
        return "❌ No project selected"

    project = project_manager.load_project(name)

    if project:
        info = project.get_info()
        summary = f"""
## Project: {info['name']}

**Description:** {info['description'] or 'No description'}

**Created:** {info['created_at'][:10]}
**Updated:** {info['updated_at'][:10]}

**Base Model:** {info['base_model']}

**Data Status:**
- RAG Database: {'✅ Available' if info['has_rag_data'] else '❌ Not created'}
- SFT Training Data: {'✅ Available' if info['has_sft_data'] else '❌ Not created'}
- Checkpoints: {info['num_checkpoints']}
- Final Model: {'✅ Available' if info['has_final_model'] else '❌ Not trained'}
"""
        return summary
    else:
        return f"❌ Failed to load project '{name}'"


def refresh_projects():
    """Refresh project list."""
    projects = project_manager.list_projects()
    return gr.update(choices=projects, value=projects[0] if projects else None)


# Build Gradio UI
with gr.Blocks(title="Custom AI Training", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🤖 Custom AI Training App

    Train small language models with custom data for educational purposes.
    """)

    # Hardware Info Section
    with gr.Accordion("💻 Hardware Information", open=False):
        hardware_output = gr.Textbox(
            label="Detected Hardware",
            lines=12,
            interactive=False,
            value=detect_hardware()
        )
        gr.Button("🔄 Refresh Hardware Info").click(
            fn=detect_hardware,
            outputs=hardware_output
        )

    # Project Management Section
    with gr.Row():
        with gr.Column(scale=2):
            project_dropdown = gr.Dropdown(
                label="Current Project",
                choices=project_manager.list_projects(),
                value=None,
                interactive=True,
                allow_custom_value=False
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh Projects")

    project_info = gr.Markdown("Select or create a project to get started.")

    # Project info on selection
    project_dropdown.change(
        fn=load_project,
        inputs=project_dropdown,
        outputs=project_info
    )

    refresh_btn.click(
        fn=refresh_projects,
        outputs=project_dropdown
    )

    # New Project Section
    with gr.Accordion("➕ Create New Project", open=False):
        with gr.Row():
            new_project_name = gr.Textbox(
                label="Project Name",
                placeholder="my-school-assistant",
                scale=2
            )
            new_project_desc = gr.Textbox(
                label="Description (optional)",
                placeholder="AI assistant for 5th grade mathematics",
                scale=3
            )
        create_btn = gr.Button("Create Project", variant="primary")
        create_output = gr.Textbox(label="Status", interactive=False)

        create_btn.click(
            fn=create_new_project,
            inputs=[new_project_name, new_project_desc],
            outputs=[create_output, project_dropdown]
        )

    gr.Markdown("---")

    # Main Tabs
    with gr.Tabs():
        # Tab 1: RAG Data Preparation
        with gr.Tab("📚 RAG Data"):
            rag_components = create_rag_tab(project_manager)

        # Tab 2: Model Selection
        with gr.Tab("🎯 Model Selection"):
            model_components = create_model_tab(project_manager)

        # Tab 3: SFT Data Generation
        with gr.Tab("✨ SFT Data"):
            sft_components = create_sft_tab(project_manager)

        # Tab 4: Training
        with gr.Tab("🔥 Training"):
            training_components = create_training_tab(project_manager)

        # Tab 5: Settings
        with gr.Tab("⚙️ Settings"):
            settings_components = create_settings_tab()

    gr.Markdown("""
    ---

    **Custom AI Training App** | Built with Gradio & Unsloth

    Phase 1: Project Setup & Hardware Detection ✅
    Phase 2: RAG Data Processing ✅
    Phase 3: Model Selection (Hugging Face) ✅
    Phase 4: SFT Data Generation (OpenRouter/Ollama) ✅
    Phase 5: Training & Export ✅
    Phase 6: Settings & Polish ✅
    """)


if __name__ == "__main__":
    print("🚀 Starting Custom AI Training App...")
    print("\n" + "="*60)

    # Show hardware info on startup
    hw_info = HardwareDetector.detect()
    print(hw_info.get_summary())

    print("="*60)
    print("\n📱 Launching Gradio UI...\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
