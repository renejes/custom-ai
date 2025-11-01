"""
Project Management Tab UI
Create, select, and manage training projects with custom locations.
"""

import gradio as gr
import os
from pathlib import Path
from typing import Optional
from core.project_manager import ProjectManager


def create_project_tab(project_manager: ProjectManager):
    """
    Create project management tab.

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    def get_projects_list() -> list:
        """Get list of all projects."""
        projects = project_manager.list_projects()
        return projects if projects else ["No projects found"]

    def create_new_project(name: str, description: str, custom_location: str, use_custom: bool) -> tuple:
        """Create a new project."""
        if not name or not name.strip():
            return (
                gr.update(),
                "Error: Please provide a project name",
                gr.update(),
                ""
            )

        name = name.strip()

        # Validate custom location if selected
        location = None
        if use_custom and custom_location and custom_location.strip():
            location = custom_location.strip()
            # Check if path exists or can be created
            try:
                Path(location).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return (
                    gr.update(),
                    f"Error: Invalid location: {str(e)}",
                    gr.update(),
                    ""
                )

        # Create project
        project = project_manager.create_project(name, description)

        if project:
            # If custom location, move project there
            if location:
                try:
                    import shutil
                    old_path = project.project_path
                    new_path = Path(location) / name
                    if old_path != new_path:
                        shutil.move(str(old_path), str(new_path))
                        # Reload project at new location
                        project_manager.current_project = None
                        project_manager.load_project(name)
                except Exception as e:
                    return (
                        gr.update(),
                        f"Warning: Project created but move failed: {str(e)}",
                        gr.update(),
                        ""
                    )

            projects_list = get_projects_list()
            project_info = get_project_details(name)

            return (
                gr.update(choices=projects_list, value=name),
                f"Success: Project '{name}' created successfully!",
                gr.update(value=project_info),
                get_project_structure(name)
            )
        else:
            return (
                gr.update(),
                f"Error: Project '{name}' already exists or invalid name",
                gr.update(),
                ""
            )

    def select_project(project_name: str) -> tuple:
        """Select a project."""
        if not project_name or project_name == "No projects found":
            return (
                "No project selected",
                ""
            )

        project = project_manager.load_project(project_name)

        if project:
            details = get_project_details(project_name)
            structure = get_project_structure(project_name)
            return (details, structure)
        else:
            return (
                f"Error: Could not load project '{project_name}'",
                ""
            )

    def get_project_details(project_name: str) -> str:
        """Get project details as markdown."""
        project = project_manager.get_current_project()

        if not project or project.name != project_name:
            project = project_manager.load_project(project_name)

        if not project:
            return "No project selected"

        info = project.get_info()

        # Count databases
        num_databases = 0
        if project.databases_path.exists():
            num_databases = len(list(project.databases_path.glob("*.db")))

        # Check CPT model
        has_cpt = project.cpt_model_path.exists() and any(project.cpt_model_path.glob("*.safetensors"))

        # Check SFT model
        has_sft = project.sft_model_path.exists() and any(project.sft_model_path.glob("*.safetensors"))

        details = f"""
**Project:** {info['name']}
**Description:** {info['description']}
**Location:** `{project.project_path}`

**Created:** {info['created_at'][:10]}
**Last Updated:** {info['updated_at'][:10]}

---

**Training Status:**
- RAG Databases: {num_databases}
- CPT Model: {"Trained" if has_cpt else "Not trained"}
- SFT Data: {"Generated" if info['has_sft_data'] else "Not generated"}
- SFT Model: {"Trained" if has_sft else "Not trained"}
- Checkpoints: {info['num_checkpoints']}
- Final Models: {"Yes" if info['has_final_model'] else "No"}
"""
        return details

    def get_project_structure(project_name: str) -> str:
        """Get project folder structure."""
        project = project_manager.get_current_project()

        if not project or project.name != project_name:
            project = project_manager.load_project(project_name)

        if not project:
            return ""

        structure = f"""
Project Folder Structure:

{project.project_path}/
├── data/
│   ├── databases/        (RAG databases)
│   └── sft_data.jsonl    (Generated Q&A pairs)
├── models/
│   ├── cpt_model/        (CPT trained model)
│   ├── sft_model/        (SFT trained model)
│   ├── checkpoints/      (Training checkpoints)
│   └── final/            (Exported models)
└── logs/                 (Training logs)
"""
        return structure

    def delete_project_fn(project_name: str, confirm_name: str) -> tuple:
        """Delete a project with confirmation."""
        if not project_name or project_name == "No projects found":
            return (
                gr.update(),
                "Error: No project selected"
            )

        if confirm_name != project_name:
            return (
                gr.update(),
                "Error: Project name does not match. Deletion cancelled."
            )

        if project_manager.delete_project(project_name):
            projects_list = get_projects_list()
            return (
                gr.update(choices=projects_list, value=None),
                f"Success: Project '{project_name}' deleted"
            )
        else:
            return (
                gr.update(),
                f"Error: Could not delete project '{project_name}'"
            )

    def open_project_folder(project_name: str) -> str:
        """Open project folder in file manager."""
        if not project_name or project_name == "No projects found":
            return "Error: No project selected"

        project = project_manager.get_current_project()
        if not project or project.name != project_name:
            project = project_manager.load_project(project_name)

        if not project:
            return f"Error: Could not load project '{project_name}'"

        try:
            import subprocess
            import platform

            project_path = str(project.project_path)

            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", project_path])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", project_path])
            else:  # Linux
                subprocess.run(["xdg-open", project_path])

            return f"Opened folder: {project_path}"
        except Exception as e:
            return f"Error opening folder: {str(e)}\n\nPath: {project_path}"

    # Build UI
    with gr.Column():
        gr.Markdown("""
        # Project Management

        Create and manage your AI training projects. Each project has its own folder with organized subfolders for data, models, and logs.
        """)

        # Current Project Section
        with gr.Accordion("Active Project", open=True):
            project_dropdown = gr.Dropdown(
                label="Select Project",
                choices=get_projects_list(),
                value=None,
                interactive=True,
                info="Choose a project to work with"
            )

            with gr.Row():
                refresh_btn = gr.Button("Refresh List", size="sm", scale=1)
                open_folder_btn = gr.Button("Open Folder", size="sm", scale=1)

            project_details = gr.Markdown("No project selected")

        gr.Markdown("---")

        # Create New Project
        with gr.Accordion("Create New Project", open=False):
            gr.Markdown("**Set up a new training project**")

            project_name_input = gr.Textbox(
                label="Project Name",
                placeholder="e.g., 'medical-chatbot', 'code-assistant'",
                info="Use a descriptive name without special characters"
            )

            project_description = gr.Textbox(
                label="Description (optional)",
                placeholder="Brief description of this project",
                lines=2
            )

            use_custom_location = gr.Checkbox(
                label="Use custom location",
                value=False,
                info="Store project in a specific folder instead of default 'projects/' directory"
            )

            custom_location_input = gr.Textbox(
                label="Custom Location",
                placeholder="/path/to/your/projects",
                visible=False,
                info="Full path where the project folder will be created"
            )

            create_btn = gr.Button("Create Project", variant="primary")
            create_status = gr.Textbox(label="Status", interactive=False, lines=2)

        # Project Structure Preview
        with gr.Accordion("Folder Structure", open=False):
            structure_display = gr.Textbox(
                label="Project Folder Structure",
                value="Select a project to view its structure",
                interactive=False,
                lines=12
            )

        # Delete Project
        gr.Markdown("---")
        with gr.Accordion("Delete Project", open=False):
            gr.Markdown("**⚠️ Warning: This action cannot be undone!**")

            delete_confirm_name = gr.Textbox(
                label="Type project name to confirm deletion",
                placeholder="Enter exact project name"
            )

            delete_btn = gr.Button("Delete Project", variant="stop")
            delete_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # Wire up events
        use_custom_location.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom_location],
            outputs=[custom_location_input]
        )

        create_btn.click(
            fn=create_new_project,
            inputs=[project_name_input, project_description, custom_location_input, use_custom_location],
            outputs=[project_dropdown, create_status, project_details, structure_display]
        )

        project_dropdown.change(
            fn=select_project,
            inputs=[project_dropdown],
            outputs=[project_details, structure_display]
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=get_projects_list()),
            outputs=[project_dropdown]
        )

        open_folder_btn.click(
            fn=open_project_folder,
            inputs=[project_dropdown],
            outputs=[create_status]
        )

        delete_btn.click(
            fn=delete_project_fn,
            inputs=[project_dropdown, delete_confirm_name],
            outputs=[project_dropdown, delete_status]
        )

    return {
        "project_dropdown": project_dropdown,
        "project_details": project_details
    }
