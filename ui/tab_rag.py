"""
RAG Data Preparation Tab UI
Handles document upload, processing, database import/export.
Supports: PDF, TXT, MD, DOCX, HTML, IPYNB, CSV, XLSX, JSON
"""

import gradio as gr
import os
import subprocess
import platform
from pathlib import Path
from typing import List, Optional
from core.rag_processor import RAGProcessor
from core.project_manager import ProjectManager


def create_rag_tab(project_manager: ProjectManager):
    """
    Create RAG data preparation tab.

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    # State variables
    uploaded_files_state = gr.State([])

    def upload_files(files) -> tuple:
        """Handle file uploads."""
        if not files:
            return [], "No files uploaded", gr.update()

        # Store uploaded files
        file_data = []
        for file in files:
            file_data.append({
                "name": os.path.basename(file.name),
                "path": file.name,
                "status": "Uploaded",
                "chunks": 0
            })

        # Create table data
        table_data = [[f["name"], f["status"]] for f in file_data]

        message = f"✅ Uploaded {len(files)} file(s)"

        return file_data, message, gr.update(value=table_data)

    def process_documents(uploaded_files, chunk_size, chunk_overlap, overwrite) -> tuple:
        """Process uploaded documents."""
        project = project_manager.get_current_project()

        if not project:
            return uploaded_files, "❌ Error: No project selected. Please create or select a project first.", gr.update(), gr.update()

        if not uploaded_files:
            return uploaded_files, "❌ Error: No files uploaded. Please upload documents first.", gr.update(), gr.update()

        # Initialize RAG processor
        db_path = str(project.get_rag_db_path())
        processor = RAGProcessor(
            db_path=db_path,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap)
        )

        # Process files
        file_paths = [f["path"] for f in uploaded_files]
        results = processor.process_files(file_paths, overwrite=overwrite)

        # Update file data with results
        updated_files = []
        for file_info in uploaded_files:
            # Find result for this file
            detail = next((d for d in results["details"] if d["filename"] == file_info["name"]), None)

            if detail:
                file_info["status"] = "✅ Processed" if detail["success"] else "❌ Failed"
                file_info["chunks"] = detail["chunks"]
            updated_files.append(file_info)

        # Create table data
        table_data = [[f["name"], f["status"]] for f in updated_files]

        # Create log message
        log_lines = [
            f"📊 Processing Results:",
            f"   Total files: {results['total_files']}",
            f"   Successful: {results['successful']}",
            f"   Failed: {results['failed']}",
            f"   Total chunks: {results['total_chunks']}",
            "",
            "📝 Details:"
        ]

        for detail in results["details"]:
            status_icon = "✅" if detail["success"] else "❌"
            log_lines.append(f"   {status_icon} {detail['filename']}: {detail['message']}")

        # Get updated stats
        stats = processor.get_stats()
        log_lines.extend([
            "",
            "📈 Database Stats:",
            f"   Total documents: {stats['total_documents']}",
            f"   Total chunks: {stats['total_chunks']}",
            f"   Database size: {stats['database_size_mb']} MB"
        ])

        # Update project config
        project.update_config({
            "rag": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "documents_processed": stats['total_documents'],
                "total_chunks": stats['total_chunks']
            }
        })

        log_message = "\n".join(log_lines)

        return updated_files, log_message, gr.update(value=table_data), gr.update(value=get_stats_summary(project))

    def get_stats_summary(project) -> str:
        """Get current database statistics."""
        if not project:
            return "No project selected"

        db_path = str(project.get_rag_db_path())

        if not os.path.exists(db_path):
            return "No database created yet. Upload and process documents to get started."

        processor = RAGProcessor(db_path)
        stats = processor.get_stats()

        summary = f"""
**Database Statistics:**
- Documents: {stats['total_documents']}
- Chunks: {stats['total_chunks']}
- Size: {stats['database_size_mb']} MB
- Path: `{stats['database_path']}`
"""
        return summary

    def export_database() -> str:
        """Export database and open file location."""
        project = project_manager.get_current_project()

        if not project:
            return "❌ Error: No project selected"

        db_path = project.get_rag_db_path()

        if not os.path.exists(db_path):
            return "❌ Error: No database found. Process documents first."

        # Open file manager at database location
        try:
            db_dir = os.path.dirname(db_path)

            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", db_dir])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", db_dir])
            else:  # Linux
                subprocess.run(["xdg-open", db_dir])

            return f"✅ Database exported!\n\nLocation: {db_path}\n\nFile manager opened at project data folder."
        except Exception as e:
            return f"✅ Database ready for export!\n\nLocation: {db_path}\n\n(Could not open file manager: {e})"

    def clear_uploads(uploaded_files) -> tuple:
        """Clear uploaded files."""
        return [], "Uploads cleared", gr.update(value=[])

    def import_database(db_file, import_mode) -> str:
        """Import database from file."""
        project = project_manager.get_current_project()

        if not project:
            return "❌ Error: No project selected"

        if not db_file:
            return "❌ Error: No database file selected"

        db_path = str(project.get_rag_db_path())
        processor = RAGProcessor(db_path)

        mode = "merge" if import_mode == "Merge" else "replace"
        success, message, num_chunks = processor.import_database(db_file.name, mode=mode)

        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"

    def export_to_format(export_format) -> str:
        """Export database to selected format."""
        project = project_manager.get_current_project()

        if not project:
            return "❌ Error: No project selected"

        db_path = str(project.get_rag_db_path())

        if not os.path.exists(db_path):
            return "❌ Error: No database found. Process documents first."

        processor = RAGProcessor(db_path)

        try:
            if export_format == "SQLite (.db)":
                # Open file manager
                db_dir = os.path.dirname(db_path)
                if platform.system() == "Darwin":
                    subprocess.run(["open", db_dir])
                elif platform.system() == "Windows":
                    subprocess.run(["explorer", db_dir])
                else:
                    subprocess.run(["xdg-open", db_dir])
                return f"✅ Database location: {db_path}\n\nFile manager opened."

            elif export_format == "JSON":
                output_path = db_path.replace('.db', '_export.json')
                success, message = processor.export_to_json(output_path)
                if success:
                    # Open file manager
                    output_dir = os.path.dirname(output_path)
                    if platform.system() == "Darwin":
                        subprocess.run(["open", output_dir])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", output_dir])
                    else:
                        subprocess.run(["xdg-open", output_dir])
                return f"✅ {message}" if success else f"❌ {message}"

            elif export_format == "JSONL":
                output_path = db_path.replace('.db', '_export.jsonl')
                success, message = processor.export_to_jsonl(output_path)
                if success:
                    # Open file manager
                    output_dir = os.path.dirname(output_path)
                    if platform.system() == "Darwin":
                        subprocess.run(["open", output_dir])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", output_dir])
                    else:
                        subprocess.run(["xdg-open", output_dir])
                return f"✅ {message}" if success else f"❌ {message}"

        except Exception as e:
            return f"❌ Error: {e}"

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## 📚 RAG Data Preparation

        Upload documents to create a knowledge base for your model.

        **Supported formats:** PDF, TXT, Markdown, DOCX, HTML, Jupyter Notebooks (.ipynb), CSV, Excel (.xlsx), JSON

        Documents will be chunked and stored in a SQLite database.
        """)

        # Upload section
        with gr.Row():
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".ipynb", ".csv", ".xlsx", ".json"],
                interactive=True
            )

        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False,
            lines=1
        )

        # Document table
        document_table = gr.Dataframe(
            headers=["Filename", "Status"],
            label="Uploaded Documents",
            interactive=False,
            wrap=True
        )

        # Processing settings
        with gr.Accordion("⚙️ Processing Settings", open=False):
            with gr.Row():
                chunk_size = gr.Number(
                    label="Chunk Size (characters)",
                    value=512,
                    minimum=100,
                    maximum=2000,
                    step=50,
                    info="Number of characters per chunk"
                )
                chunk_overlap = gr.Number(
                    label="Chunk Overlap (characters)",
                    value=50,
                    minimum=0,
                    maximum=500,
                    step=10,
                    info="Overlap between consecutive chunks"
                )

            overwrite = gr.Checkbox(
                label="Overwrite existing documents",
                value=False,
                info="If checked, reprocess documents that are already in the database"
            )

        # Action buttons
        with gr.Row():
            process_btn = gr.Button("🔄 Process Documents", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Clear Uploads", scale=1)

        # Processing log
        processing_log = gr.Textbox(
            label="Processing Log",
            lines=10,
            interactive=False,
            placeholder="Upload and process documents to see logs here..."
        )

        # Import/Export Section
        gr.Markdown("---")
        gr.Markdown("### 📥 Import / 📤 Export")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Import Database**")
                db_import_file = gr.File(
                    label="Select Database File (.db)",
                    file_types=[".db"],
                    file_count="single"
                )
                import_mode = gr.Radio(
                    label="Import Mode",
                    choices=["Merge", "Replace"],
                    value="Merge",
                    info="Merge: Add to existing | Replace: Delete existing first"
                )
                import_btn = gr.Button("📥 Import Database", variant="secondary")
                import_output = gr.Textbox(label="Import Status", lines=2, interactive=False)

            with gr.Column():
                gr.Markdown("**Export Database**")
                export_format = gr.Radio(
                    label="Export Format",
                    choices=["SQLite (.db)", "JSON", "JSONL"],
                    value="SQLite (.db)",
                    info="SQLite: Full database | JSON/JSONL: For other RAG systems"
                )
                export_format_btn = gr.Button("📤 Export", variant="secondary")
                export_output = gr.Textbox(label="Export Status", lines=2, interactive=False)

        # Database stats
        gr.Markdown("---")
        with gr.Accordion("📊 Database Statistics", open=False):
            stats_display = gr.Markdown(get_stats_summary(project_manager.get_current_project()))
            refresh_stats_btn = gr.Button("🔄 Refresh Statistics")

        # Wire up events
        file_upload.change(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[uploaded_files_state, upload_status, document_table]
        )

        process_btn.click(
            fn=process_documents,
            inputs=[uploaded_files_state, chunk_size, chunk_overlap, overwrite],
            outputs=[uploaded_files_state, processing_log, document_table, stats_display]
        )

        clear_btn.click(
            fn=clear_uploads,
            inputs=[uploaded_files_state],
            outputs=[uploaded_files_state, upload_status, document_table]
        )

        import_btn.click(
            fn=import_database,
            inputs=[db_import_file, import_mode],
            outputs=[import_output]
        )

        export_format_btn.click(
            fn=export_to_format,
            inputs=[export_format],
            outputs=[export_output]
        )

        refresh_stats_btn.click(
            fn=lambda: get_stats_summary(project_manager.get_current_project()),
            outputs=[stats_display]
        )

    return {
        "uploaded_files_state": uploaded_files_state,
        "stats_display": stats_display
    }
