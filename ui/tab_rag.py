"""
RAG Data Preparation Tab UI
Handles document upload, processing, database import/export.
Supports: PDF, TXT, MD, DOCX, HTML, IPYNB, CSV, XLSX, JSON
Multi-Database Management.
"""

import gradio as gr
import os
import subprocess
import platform
from pathlib import Path
from typing import List, Optional, Tuple
from core.rag_processor import RAGProcessor
from core.project_manager import ProjectManager
from utils.config import get_global_config


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
    config = get_global_config()

    # Database Management Functions
    def get_database_list() -> List[str]:
        """Get list of all databases."""
        databases = list(config.get_rag_databases().keys())
        if "wissensbasis" not in databases:
            databases.insert(0, "wissensbasis")
        return databases

    def get_database_path(db_name: str) -> str:
        """Get path for a database."""
        project = project_manager.get_current_project()
        if not project:
            return ""

        # All databases now stored in databases/ subfolder
        return str(project.databases_path / f"{db_name}.db")

    def create_database(db_name: str) -> tuple:
        """Create a new database."""
        if not db_name or db_name.strip() == "":
            return (
                gr.update(),
                "❌ Error: Please provide a database name",
                gr.update()
            )

        db_name = db_name.strip()

        # Check if already exists
        existing = get_database_list()
        if db_name in existing:
            return (
                gr.update(),
                f"❌ Error: Database '{db_name}' already exists",
                gr.update()
            )

        # Create database path
        db_path = get_database_path(db_name)

        # Add to config
        if db_name != "wissensbasis":
            config.add_rag_database(db_name, db_path)
            config.save()

        # Create empty database
        processor = RAGProcessor(db_path)
        stats = processor.get_stats()

        # Update dropdown
        db_list = get_database_list()

        return (
            gr.update(choices=db_list, value=db_name),
            f"✅ Database '{db_name}' created successfully!",
            gr.update(choices=db_list)
        )

    def refresh_databases() -> tuple:
        """Refresh database list."""
        db_list = get_database_list()
        active_db = config.get_active_rag_database()

        if active_db not in db_list:
            active_db = "wissensbasis"

        return (
            gr.update(choices=db_list, value=active_db),
            gr.update(choices=db_list)
        )

    def select_database(db_name: str) -> str:
        """Select active database."""
        if not db_name:
            return "❌ No database selected"

        config.set_active_rag_database(db_name)
        config.save()

        db_path = get_database_path(db_name)
        if os.path.exists(db_path):
            processor = RAGProcessor(db_path)
            stats = processor.get_stats()
            return f"✅ Active database: '{db_name}'\n📊 Documents: {stats['total_documents']}, Chunks: {stats['total_chunks']}"
        else:
            return f"✅ Active database: '{db_name}' (empty)"

    def merge_databases_fn(source_dbs: List[str], target_name: str) -> str:
        """Merge multiple databases."""
        if not source_dbs or len(source_dbs) < 2:
            return "Error: Please select at least 2 databases to merge"

        if not target_name or target_name.strip() == "":
            return "Error: Please provide a target database name"

        target_name = target_name.strip()

        # Get source paths
        source_paths = [get_database_path(db) for db in source_dbs]

        # Check if sources exist
        missing = [db for db, path in zip(source_dbs, source_paths) if not os.path.exists(path)]
        if missing:
            return f"Error: Databases not found: {', '.join(missing)}"

        # Create target path
        target_path = get_database_path(target_name)

        # Merge
        processor = RAGProcessor(target_path)
        success, message, chunks = processor.merge_databases(source_paths, target_path)

        if success:
            # Add to config
            if target_name != "wissensbasis":
                config.add_rag_database(target_name, target_path)
                config.save()

            return f"{message}"
        else:
            return f"{message}"

    def import_database_simple(db_file, db_name: str) -> str:
        """Import a database file as a new database."""
        if not db_file:
            return "Error: Please select a database file"

        if not db_name or db_name.strip() == "":
            return "Error: Please provide a name for the imported database"

        db_name = db_name.strip()

        # Check if name already exists
        existing = get_database_list()
        if db_name in existing:
            return f"Error: Database '{db_name}' already exists"

        try:
            # Get target path
            target_path = get_database_path(db_name)

            # Copy database file
            import shutil
            shutil.copy(db_file, target_path)

            # Add to config
            if db_name != "wissensbasis":
                config.add_rag_database(db_name, target_path)
                config.save()

            # Get stats
            processor = RAGProcessor(target_path)
            stats = processor.get_stats()

            return f"Success: Database '{db_name}' imported!\n\nDocuments: {stats['total_documents']}\nChunks: {stats['total_chunks']}"

        except Exception as e:
            return f"Error importing database: {str(e)}"

    def load_database_chunks(page: int = 1) -> Tuple[list, str, str, int]:
        """Load chunks from active database for viewing with pagination."""
        project = project_manager.get_current_project()

        if not project:
            return [], "No project selected", "Page 0 of 0", 1

        # Get active database
        active_db = config.get_active_rag_database()
        db_path = get_database_path(active_db)

        if not os.path.exists(db_path):
            return [], f"Database '{active_db}' not found", "Page 0 of 0", 1

        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_chunks = cursor.fetchone()[0]

            if total_chunks == 0:
                conn.close()
                return [], f"Database '{active_db}' is empty", "Page 0 of 0", 1

            # Calculate pagination
            chunks_per_page = 1000
            total_pages = (total_chunks + chunks_per_page - 1) // chunks_per_page  # Ceiling division
            page = max(1, min(page, total_pages))  # Clamp to valid range
            offset = (page - 1) * chunks_per_page

            # Get chunks for current page
            cursor.execute("""
                SELECT id, filename, chunk_index, chunk_text
                FROM documents
                ORDER BY filename, chunk_index
                LIMIT ? OFFSET ?
            """, (chunks_per_page, offset))

            rows = cursor.fetchall()
            conn.close()

            # Format for DataFrame: [ID, Filename, Chunk Index, Preview]
            table_data = []
            for row in rows:
                chunk_id, filename, chunk_index, chunk_text = row
                preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                table_data.append([chunk_id, filename, chunk_index, preview])

            # Create status and page info
            status = f"Loaded {len(rows)} chunks from '{active_db}' (Total: {total_chunks})"
            page_info_text = f"Page {page} of {total_pages} (Chunks {offset + 1}-{offset + len(rows)} of {total_chunks})"

            return table_data, status, page_info_text, page

        except Exception as e:
            return [], f"Error loading chunks: {str(e)}", "Page 0 of 0", 1

    def load_next_page(current_pg: int) -> Tuple[list, str, str, int]:
        """Load next page of chunks."""
        return load_database_chunks(current_pg + 1)

    def load_prev_page(current_pg: int) -> Tuple[list, str, str, int]:
        """Load previous page of chunks."""
        return load_database_chunks(max(1, current_pg - 1))

    def goto_page(page_num: int) -> Tuple[list, str, str, int]:
        """Go to specific page."""
        return load_database_chunks(int(page_num))

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

        message = f" Uploaded {len(files)} file(s)"

        return file_data, message, gr.update(value=table_data)

    def process_documents(uploaded_files, use_semantic, embedding_model, chunk_size, chunk_overlap, overwrite) -> tuple:
        """Process uploaded documents."""
        project = project_manager.get_current_project()

        if not project:
            return uploaded_files, "Error: No project selected. Please create or select a project first.", gr.update(), gr.update()

        if not uploaded_files:
            return uploaded_files, "Error: No files uploaded. Please upload documents first.", gr.update(), gr.update()

        # Get active database
        active_db = config.get_active_rag_database()
        db_path = get_database_path(active_db)

        # Initialize RAG processor with semantic chunking
        processor = RAGProcessor(
            db_path=db_path,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            use_semantic_chunking=use_semantic,
            embedding_model=embedding_model
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
                file_info["status"] = " Processed" if detail["success"] else " Failed"
                file_info["chunks"] = detail["chunks"]
            updated_files.append(file_info)

        # Create table data
        table_data = [[f["name"], f["status"]] for f in updated_files]

        # Create log message
        log_lines = [
            f" Processing Results:",
            f"   Total files: {results['total_files']}",
            f"   Successful: {results['successful']}",
            f"   Failed: {results['failed']}",
            f"   Total chunks: {results['total_chunks']}",
            "",
            " Details:"
        ]

        for detail in results["details"]:
            status_icon = "" if detail["success"] else ""
            log_lines.append(f"   {status_icon} {detail['filename']}: {detail['message']}")

        # Get updated stats
        stats = processor.get_stats()
        log_lines.extend([
            "",
            " Database Stats:",
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
            return " Error: No project selected"

        db_path = project.get_rag_db_path()

        if not os.path.exists(db_path):
            return " Error: No database found. Process documents first."

        # Open file manager at database location
        try:
            db_dir = os.path.dirname(db_path)

            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", db_dir])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", db_dir])
            else:  # Linux
                subprocess.run(["xdg-open", db_dir])

            return f" Database exported!\n\nLocation: {db_path}\n\nFile manager opened at project data folder."
        except Exception as e:
            return f" Database ready for export!\n\nLocation: {db_path}\n\n(Could not open file manager: {e})"

    def clear_uploads(uploaded_files) -> tuple:
        """Clear uploaded files."""
        return [], "Uploads cleared", gr.update(value=[])

    def import_database(db_file, import_mode) -> str:
        """Import database from file."""
        project = project_manager.get_current_project()

        if not project:
            return " Error: No project selected"

        if not db_file:
            return " Error: No database file selected"

        db_path = str(project.get_rag_db_path())
        processor = RAGProcessor(db_path)

        mode = "merge" if import_mode == "Merge" else "replace"
        success, message, num_chunks = processor.import_database(db_file.name, mode=mode)

        if success:
            return f" {message}"
        else:
            return f" {message}"

    def export_to_format(export_format) -> str:
        """Export database to selected format."""
        project = project_manager.get_current_project()

        if not project:
            return " Error: No project selected"

        db_path = str(project.get_rag_db_path())

        if not os.path.exists(db_path):
            return " Error: No database found. Process documents first."

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
                return f" Database location: {db_path}\n\nFile manager opened."

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
                return f" {message}" if success else f" {message}"

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
                return f" {message}" if success else f" {message}"
            else:
                return f" Error: Unknown export format: {export_format}"

        except Exception as e:
            return f" Error: {e}"

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ##  RAG Data Preparation

        Upload documents to create a knowledge base for your model.

        **Supported formats:** PDF, TXT, Markdown, DOCX, HTML, Jupyter Notebooks (.ipynb), CSV, Excel (.xlsx), JSON

        Documents will be chunked and stored in named SQLite databases.
        """)

        # Step 1: Database Management
        with gr.Accordion("📊 Step 1: Database Management", open=True):
            gr.Markdown("**Manage multiple knowledge bases**")

            with gr.Row():
                with gr.Column(scale=2):
                    database_dropdown = gr.Dropdown(
                        label="Active Database",
                        choices=["wissensbasis"],
                        value="wissensbasis",
                        interactive=True,
                        info="Select or create a database"
                    )
                with gr.Column(scale=1):
                    refresh_db_btn = gr.Button("🔄 Refresh", size="sm")

            with gr.Row():
                new_db_name = gr.Textbox(
                    label="New Database Name",
                    placeholder="e.g., 'python_docs', 'medical_kb'",
                    scale=2
                )
                create_db_btn = gr.Button("➕ Create Database", variant="primary", scale=1)

            db_status = gr.Textbox(label="Database Status", interactive=False, lines=2)

            # Merge databases
            with gr.Accordion("🔀 Merge Databases", open=False):
                gr.Markdown("**Combine multiple databases into one**")
                merge_source_dbs = gr.CheckboxGroup(
                    label="Select databases to merge",
                    choices=["wissensbasis"],
                    value=[]
                )
                merge_target_name = gr.Textbox(
                    label="Target database name",
                    placeholder="e.g., 'merged_kb'"
                )
                merge_btn = gr.Button("🔀 Merge Selected Databases", variant="secondary")
                merge_output = gr.Textbox(label="Merge Status", interactive=False, lines=2)

        gr.Markdown("---")

        # Step 2: Upload Documents
        with gr.Accordion("📁 Step 2: Upload Documents", open=True):
            gr.Markdown("**Upload files to the active database**")

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
            with gr.Accordion("Processing Settings", open=True):
                gr.Markdown("**Chunking Strategy**")

                use_semantic_chunking = gr.Checkbox(
                    label="Use Semantic Chunking (Recommended for German)",
                    value=True,
                    info="Split text based on meaning instead of fixed size - better context preservation"
                )

                embedding_model_choice = gr.Radio(
                    label="Embedding Model",
                    choices=[
                        ("Multilingual (50+ languages, fast) - RECOMMENDED", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        ("Google EmbeddingGemma (100+ languages, slower)", "google/embeddinggemma-300m"),
                    ],
                    value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    info="Model for semantic similarity detection"
                )

                gr.Markdown("**Advanced Settings**")

                with gr.Row():
                    chunk_size = gr.Number(
                        label="Max Chunk Size (characters)",
                        value=1000,
                        minimum=100,
                        maximum=4000,
                        step=100,
                        info="Maximum size for semantic chunks (fallback limit)"
                    )
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap (characters)",
                        value=100,
                        minimum=0,
                        maximum=500,
                        step=50,
                        info="Overlap between consecutive chunks"
                    )

                overwrite = gr.Checkbox(
                    label="Overwrite existing documents",
                    value=False,
                    info="If checked, reprocess documents that are already in the database"
                )

            # Action buttons
            with gr.Row():
                process_btn = gr.Button(" Process Documents", variant="primary", scale=2)
                clear_btn = gr.Button(" Clear Uploads", scale=1)

            # Processing log
            processing_log = gr.Textbox(
                label="Processing Log",
                lines=10,
                interactive=False,
                placeholder="Upload and process documents to see logs here..."
            )

        # Import/Export Section
        gr.Markdown("---")
        gr.Markdown("### Import / Export")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Import Database**")
                db_import_file = gr.File(
                    label="Select Database File (.db)",
                    file_types=[".db"],
                    file_count="single"
                )
                import_db_name = gr.Textbox(
                    label="Database Name",
                    placeholder="e.g., 'imported_kb'",
                    info="Name for the imported database"
                )
                import_btn = gr.Button("Import as New Database", variant="secondary")
                import_output = gr.Textbox(label="Import Status", lines=2, interactive=False)

            with gr.Column():
                gr.Markdown("**Export Database**")
                export_format = gr.Radio(
                    label="Export Format",
                    choices=["SQLite (.db)", "JSON", "JSONL"],
                    value="SQLite (.db)",
                    info="SQLite: Full database | JSON/JSONL: For other RAG systems"
                )
                export_format_btn = gr.Button("Export", variant="secondary")
                export_output = gr.Textbox(label="Export Status", lines=2, interactive=False)

        # Database Viewer
        gr.Markdown("---")
        with gr.Accordion("Database Viewer", open=False):
            gr.Markdown("**Browse chunks in the active database**")

            load_db_chunks_btn = gr.Button("Load Chunks from Active Database", variant="secondary")

            with gr.Row():
                db_chunks_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    scale=3
                )
                page_info = gr.Textbox(
                    label="Page",
                    value="Page 0 of 0",
                    interactive=False,
                    scale=1
                )

            with gr.Row():
                prev_page_btn = gr.Button("◀ Previous 1000", size="sm", scale=1)
                next_page_btn = gr.Button("Next 1000 ▶", size="sm", scale=1)
                page_input = gr.Number(
                    label="Go to page",
                    value=1,
                    minimum=1,
                    step=1,
                    scale=1
                )
                goto_page_btn = gr.Button("Go", size="sm", scale=1)

            db_chunks_dataframe = gr.Dataframe(
                headers=["ID", "Filename", "Chunk Index", "Preview"],
                datatype=["number", "str", "number", "str"],
                col_count=(4, "fixed"),
                label="Database Chunks",
                interactive=False,
                wrap=True,
                value=[],
                row_count=(15, "dynamic")
            )

            # Hidden state for current page
            current_page = gr.State(1)

        # Database stats
        gr.Markdown("---")
        with gr.Accordion("Database Statistics", open=False):
            stats_display = gr.Markdown(get_stats_summary(project_manager.get_current_project()))
            refresh_stats_btn = gr.Button("Refresh Statistics")

        # Wire up events
        # Database management
        create_db_btn.click(
            fn=create_database,
            inputs=[new_db_name],
            outputs=[database_dropdown, db_status, merge_source_dbs]
        )

        refresh_db_btn.click(
            fn=refresh_databases,
            outputs=[database_dropdown, merge_source_dbs]
        )

        database_dropdown.change(
            fn=select_database,
            inputs=[database_dropdown],
            outputs=[db_status]
        )

        merge_btn.click(
            fn=merge_databases_fn,
            inputs=[merge_source_dbs, merge_target_name],
            outputs=[merge_output]
        )

        # File upload and processing
        file_upload.change(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[uploaded_files_state, upload_status, document_table]
        )

        process_btn.click(
            fn=process_documents,
            inputs=[uploaded_files_state, use_semantic_chunking, embedding_model_choice, chunk_size, chunk_overlap, overwrite],
            outputs=[uploaded_files_state, processing_log, document_table, stats_display]
        )

        clear_btn.click(
            fn=clear_uploads,
            inputs=[uploaded_files_state],
            outputs=[uploaded_files_state, upload_status, document_table]
        )

        # Import/Export
        import_btn.click(
            fn=import_database_simple,
            inputs=[db_import_file, import_db_name],
            outputs=[import_output]
        )

        export_format_btn.click(
            fn=export_to_format,
            inputs=[export_format],
            outputs=[export_output]
        )

        # Database Viewer
        load_db_chunks_btn.click(
            fn=load_database_chunks,
            outputs=[db_chunks_dataframe, db_chunks_status, page_info, current_page]
        )

        next_page_btn.click(
            fn=load_next_page,
            inputs=[current_page],
            outputs=[db_chunks_dataframe, db_chunks_status, page_info, current_page]
        )

        prev_page_btn.click(
            fn=load_prev_page,
            inputs=[current_page],
            outputs=[db_chunks_dataframe, db_chunks_status, page_info, current_page]
        )

        goto_page_btn.click(
            fn=goto_page,
            inputs=[page_input],
            outputs=[db_chunks_dataframe, db_chunks_status, page_info, current_page]
        )

        # Stats
        refresh_stats_btn.click(
            fn=lambda: get_stats_summary(project_manager.get_current_project()),
            outputs=[stats_display]
        )

    return {
        "uploaded_files_state": uploaded_files_state,
        "stats_display": stats_display
    }
