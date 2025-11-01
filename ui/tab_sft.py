"""
SFT Data Generation Tab UI (Phase 3)
Generate Q&A pairs and behavioral dialogs from RAG knowledge base using AI assistant.
"""

from __future__ import annotations

import gradio as gr
import asyncio
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
from core.sft_generator import SFTGenerator
from core.ollama_client import OllamaClient
from core.project_manager import ProjectManager
from utils.config import get_global_config


def create_sft_tab(project_manager: ProjectManager):
    """
    Create SFT data generation tab.

    This is Phase 3 of the 4-phase training pipeline:
    1. RAG Data (Tab 1) - Upload documents, create wissensbasis.sqlite
    2. CPT Training (Tab 2) - Train base model on RAG chunks
    3. SFT Data (Tab 3 - THIS TAB) - Generate Q&A pairs from RAG data
    4. SFT Training (Tab 4) - Fine-tune CPT model on Q&A pairs

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    ollama_client = OllamaClient()
    config = get_global_config()

    def check_prerequisites() -> str:
        """Check if RAG database and CPT model exist."""
        project = project_manager.get_current_project()

        if not project:
            return "Error: No project selected"

        issues = []

        # Check RAG database
        rag_db_path = project.get_rag_db_path()
        if not rag_db_path.exists():
            issues.append("RAG database not found (Tab 1)")

        # Check CPT model (optional but recommended)
        cpt_model_path = project.cpt_model_path
        if not cpt_model_path.exists():
            issues.append("CPT model not found (Tab 2) - Optional but recommended")

        if issues:
            return "Prerequisites:\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            try:
                conn = sqlite3.connect(rag_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM rag_chunks")
                num_chunks = cursor.fetchone()[0]
                conn.close()

                return f"""Ready for SFT Data Generation!

RAG Database: wissensbasis.sqlite
Total Chunks: {num_chunks}
CPT Model: Available

You can now browse chunks and generate Q&A pairs."""
            except Exception as e:
                return f"Error reading RAG database: {str(e)}"

    def load_rag_chunks() -> Tuple[List[Dict], str]:
        """Load RAG chunks from database for browsing."""
        project = project_manager.get_current_project()

        if not project:
            return [], "No project selected"

        rag_db_path = project.get_rag_db_path()

        if not rag_db_path.exists():
            return [], "RAG database not found. Process documents in Tab 1 first."

        try:
            conn = sqlite3.connect(rag_db_path)
            cursor = conn.cursor()

            # Get all chunks with metadata
            cursor.execute("""
                SELECT id, content, source, chunk_index, metadata
                FROM rag_chunks
                ORDER BY id
            """)

            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    "id": row[0],
                    "content": row[1],
                    "source": row[2],
                    "chunk_index": row[3],
                    "metadata": row[4]
                })

            conn.close()

            return chunks, f"Loaded {len(chunks)} chunks from wissensbasis.sqlite"

        except Exception as e:
            return [], f"Error loading chunks: {str(e)}"

    def load_chunks_to_dataframe() -> Tuple[list, str, str]:
        """Load chunks and format for DataFrame display."""
        chunks, msg = load_rag_chunks()

        if not chunks:
            return [], msg, "0 selected"

        # Format for DataFrame: [Select, ID, Source, Chunk, Preview]
        table_data = []
        for chunk in chunks:
            preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            table_data.append([
                False,  # Select checkbox
                chunk['id'],
                chunk['source'],
                chunk['chunk_index'],
                preview
            ])

        return table_data, msg, f"0 / {len(chunks)} selected"

    def select_all_chunks(df_data: list) -> Tuple[list, str]:
        """Select all chunks in DataFrame."""
        if not df_data:
            return df_data, "0 selected"

        for row in df_data:
            row[0] = True

        return df_data, f"{len(df_data)} / {len(df_data)} selected"

    def deselect_all_chunks(df_data: list) -> Tuple[list, str]:
        """Deselect all chunks in DataFrame."""
        if not df_data:
            return df_data, "0 selected"

        for row in df_data:
            row[0] = False

        return df_data, f"0 / {len(df_data)} selected"

    def update_selected_count(df_data: list) -> str:
        """Count selected chunks."""
        if not df_data:
            return "0 selected"

        selected = sum(1 for row in df_data if row[0])
        return f"{selected} / {len(df_data)} selected"

    def get_openrouter_models() -> List[str]:
        """Get list of available OpenRouter models."""
        return [
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3-70b-instruct",
            "meta-llama/llama-3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "google/gemini-pro"
        ]

    def refresh_ollama_models() -> Tuple[gr.update, str]:
        """Refresh Ollama models list."""
        success, models, error = ollama_client.list_models()

        if success:
            if not models:
                return gr.update(choices=["No models found"], value=None), "No Ollama models installed"

            model_names = [m["name"] for m in models]
            return gr.update(choices=model_names, value=model_names[0] if model_names else None), f"Found {len(models)} model(s)"
        else:
            return gr.update(choices=["Error"], value=None), f"Error: {error}"

    def test_api_connection(provider, api_key, ollama_url, openrouter_model, ollama_model) -> str:
        """Test API connection."""
        if provider == "OpenRouter":
            if not api_key.strip():
                return "Error: Please enter an API key"

            generator = SFTGenerator(provider="openrouter", api_key=api_key)
            success, msg = asyncio.run(generator.test_connection(openrouter_model))
            return msg

        else:  # Ollama
            generator = SFTGenerator(provider="ollama", base_url=ollama_url)
            success, msg = asyncio.run(generator.test_connection(ollama_model))
            return msg

    def generate_qa_from_chunks(
        df_data: list,
        provider: str,
        api_key: str,
        ollama_url: str,
        openrouter_model: str,
        ollama_model: str,
        num_samples: int,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float,
        qa_style: str,
        progress=gr.Progress()
    ) -> str:
        """Generate Q&A pairs from selected RAG chunks."""
        project = project_manager.get_current_project()

        if not project:
            return "❌ Error: No project selected"

        # Get selected chunks from DataFrame
        if not df_data:
            return "❌ Error: No chunks loaded. Please load chunks first."

        selected_ids = [int(row[1]) for row in df_data if row[0]]  # row[0] = Select, row[1] = ID

        if not selected_ids:
            return "❌ Error: No chunks selected. Please select at least one chunk."

        # Load full chunk data for selected IDs
        progress(0.1, desc="Loading selected RAG chunks...")
        all_chunks, msg = load_rag_chunks()

        if not all_chunks:
            return f"❌ Error: {msg}"

        # Filter to selected chunks
        chunks = [chunk for chunk in all_chunks if chunk['id'] in selected_ids]

        if not chunks:
            return "❌ Error: Could not load selected chunks"

        # Validate inputs
        if num_samples < 1:
            return "Error: Number of samples must be at least 1"

        if num_samples > len(chunks):
            return f"Error: Cannot generate {num_samples} samples from {len(chunks)} chunks"

        # Initialize generator
        if provider == "OpenRouter":
            if not api_key.strip():
                return "Error: API key is required for OpenRouter"
            generator = SFTGenerator(provider="openrouter", api_key=api_key)
            model = openrouter_model
        else:  # Ollama
            generator = SFTGenerator(provider="ollama", base_url=ollama_url)
            model = ollama_model

        # Prepare prompt based on style
        style_instructions = {
            "Factual Q&A": "Generate factual questions and accurate answers based on the content.",
            "Conversational": "Generate natural conversational exchanges like a helpful tutor.",
            "Socratic": "Generate Socratic-style questions that lead to deeper understanding.",
            "Problem-Solving": "Generate problem-based questions with step-by-step solutions.",
            "Multiple Choice": "Generate multiple choice questions with correct answers and explanations."
        }

        full_system_prompt = f"""{system_prompt}

Style: {qa_style}
{style_instructions.get(qa_style, "")}

Return response in JSON format:
{{"instruction": "the question or instruction", "output": "the response or answer"}}"""

        # Generate Q&A pairs
        progress(0.2, desc="Generating Q&A pairs...")
        generated_samples = []

        try:
            # Sample chunks evenly
            step = max(1, len(chunks) // num_samples)
            selected_chunks = chunks[::step][:num_samples]

            for i, chunk in enumerate(selected_chunks):
                progress((i, num_samples), desc=f"Generating Q&A {i+1}/{num_samples}")

                # Format user prompt with chunk content
                user_prompt = user_prompt_template.replace("{chunk}", chunk["content"])

                # Generate Q&A
                success, response, error = asyncio.run(
                    generator.generate_single_sample(
                        model=model,
                        system_prompt=full_system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature
                    )
                )

                if success and response:
                    try:
                        # Try to parse JSON response
                        qa_pair = json.loads(response)
                        if "instruction" in qa_pair and "output" in qa_pair:
                            generated_samples.append(qa_pair)
                    except json.JSONDecodeError:
                        # If not JSON, create structured format
                        generated_samples.append({
                            "instruction": f"Question based on {chunk['source']}",
                            "output": response
                        })

            # Save to JSONL
            progress(0.9, desc="Saving to JSONL...")
            sft_data_path = project.project_path / "data" / "sft_data.jsonl"

            with open(sft_data_path, 'w', encoding='utf-8') as f:
                for sample in generated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            progress(1.0, desc="Complete!")

            return f"""SFT Data Generation Complete!

Generated: {len(generated_samples)} Q&A pairs
Style: {qa_style}
Saved to: sft_data.jsonl

Next Steps:
1. Review generated data below
2. Go to Tab 4 to train SFT model with this data"""

        except Exception as e:
            return f"Error during generation: {str(e)}"

    def preview_sft_data() -> str:
        """Preview generated SFT data."""
        project = project_manager.get_current_project()

        if not project:
            return "No project selected"

        sft_data_path = project.project_path / "data" / "sft_data.jsonl"

        if not sft_data_path.exists():
            return "No SFT data found. Generate Q&A pairs first."

        try:
            samples = []
            with open(sft_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            if not samples:
                return "SFT data file is empty"

            # Show first 5 samples
            preview_lines = [f"**Total Samples:** {len(samples)}\n"]
            preview_lines.append("**Preview (first 5 samples):**\n")

            for i, sample in enumerate(samples[:5], 1):
                preview_lines.append(f"**Sample {i}:**")
                preview_lines.append(f"**Instruction:** {sample.get('instruction', 'N/A')}")
                preview_lines.append(f"**Output:** {sample.get('output', 'N/A')[:200]}...")
                preview_lines.append("---\n")

            return "\n".join(preview_lines)

        except Exception as e:
            return f"Error reading SFT data: {str(e)}"

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## Phase 3: SFT Data Generation

        Generate Q&A pairs and behavioral dialogs from your RAG knowledge base using AI assistance.

        **What happens here:**
        1. Browse chunks from wissensbasis.sqlite (same data used in CPT)
        2. Use AI assistant (Ollama/OpenRouter) to generate Q&A pairs
        3. Generate in different styles (factual, conversational, Socratic, etc.)
        4. Export as JSONL for SFT training in Tab 4
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

        # RAG Chunks Browser with DataFrame Selection
        with gr.Accordion("Browse & Select RAG Chunks", open=True):
            gr.Markdown("**View and select chunks from wissensbasis.sqlite for Q&A generation**")

            load_chunks_btn = gr.Button("Load Chunks from Database", variant="secondary")
            chunks_status = gr.Textbox(label="Status", interactive=False, lines=1)

            gr.Markdown("**Select chunks to generate Q&A pairs:**")

            chunks_dataframe = gr.Dataframe(
                headers=["Select", "ID", "Source", "Chunk", "Preview"],
                datatype=["bool", "number", "str", "number", "str"],
                col_count=(5, "fixed"),
                label="RAG Chunks",
                interactive=True,
                wrap=True,
                value=[],
                row_count=(10, "dynamic")
            )

            with gr.Row():
                select_all_btn = gr.Button("Select All", size="sm")
                deselect_all_btn = gr.Button("Deselect All", size="sm")
                selected_count_display = gr.Textbox(
                    label="Selected Chunks",
                    value="0 selected",
                    interactive=False,
                    scale=2
                )

        gr.Markdown("---")

        # AI Assistant Configuration
        gr.Markdown("### AI Assistant Configuration")

        with gr.Row():
            provider = gr.Radio(
                label="AI Provider",
                choices=["Ollama", "OpenRouter"],
                value="Ollama",
                info="Choose where to run the AI assistant"
            )

        with gr.Tabs():
            with gr.Tab("Ollama"):
                ollama_url = gr.Textbox(
                    label="Ollama Base URL",
                    value=config.get_api_config().get("ollama_base_url", "http://localhost:11434"),
                    info="URL of your Ollama instance"
                )

                with gr.Row():
                    ollama_model = gr.Dropdown(
                        label="Model",
                        choices=["llama3:8b", "phi3:mini"],
                        value="llama3:8b",
                        allow_custom_value=True
                    )
                    refresh_ollama_btn = gr.Button("Refresh", scale=0)

                ollama_status = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Tab("OpenRouter"):
                openrouter_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=config.get_api_config().get("openrouter_api_key", ""),
                    info="Get your key from openrouter.ai"
                )

                openrouter_model = gr.Dropdown(
                    label="Model",
                    choices=get_openrouter_models(),
                    value="openai/gpt-3.5-turbo"
                )

        test_connection_btn = gr.Button("Test Connection", variant="secondary")
        connection_status = gr.Textbox(label="Connection Status", interactive=False, lines=2)

        gr.Markdown("---")

        # Q&A Generation Settings
        gr.Markdown("### Q&A Generation Settings")

        prompts = config.get_sft_prompts()

        with gr.Row():
            num_samples = gr.Slider(
                label="Number of Q&A Pairs",
                minimum=1,
                maximum=500,
                value=50,
                step=1,
                info="How many Q&A pairs to generate"
            )

            qa_style = gr.Dropdown(
                label="Q&A Style",
                choices=[
                    "Factual Q&A",
                    "Conversational",
                    "Socratic",
                    "Problem-Solving",
                    "Multiple Choice"
                ],
                value="Factual Q&A",
                info="Choose the style of generated Q&A"
            )

        system_prompt = gr.Textbox(
            label="System Prompt",
            value=prompts.get("system_prompt", "You are a helpful AI assistant generating training data."),
            lines=3,
            info="Instructions for the AI assistant"
        )

        user_prompt_template = gr.Textbox(
            label="User Prompt Template",
            value=prompts.get("user_prompt_template", "Based on this content, generate a question and answer:\n\n{chunk}"),
            lines=3,
            info="Use {chunk} as placeholder for RAG content"
        )

        temperature = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            info="Higher = more creative, Lower = more focused"
        )

        # Generate button
        gr.Markdown("---")
        generate_sft_btn = gr.Button("Generate Q&A Pairs", variant="primary", size="lg")

        generation_output = gr.Textbox(
            label="Generation Log",
            lines=10,
            interactive=False,
            placeholder="Click 'Generate Q&A Pairs' to start..."
        )

        # Preview section
        with gr.Accordion("Preview Generated Data", open=False):
            preview_btn = gr.Button("Preview SFT Data", variant="secondary")
            preview_output = gr.Markdown("No data to preview")

        # Wire up events
        check_btn.click(
            fn=check_prerequisites,
            outputs=[prerequisites_output]
        )

        # Load chunks into DataFrame
        load_chunks_btn.click(
            fn=load_chunks_to_dataframe,
            outputs=[chunks_dataframe, chunks_status, selected_count_display]
        )

        # Select/Deselect all
        select_all_btn.click(
            fn=select_all_chunks,
            inputs=[chunks_dataframe],
            outputs=[chunks_dataframe, selected_count_display]
        )

        deselect_all_btn.click(
            fn=deselect_all_chunks,
            inputs=[chunks_dataframe],
            outputs=[chunks_dataframe, selected_count_display]
        )

        # Update count when DataFrame changes
        chunks_dataframe.change(
            fn=update_selected_count,
            inputs=[chunks_dataframe],
            outputs=[selected_count_display]
        )

        # Ollama refresh
        refresh_ollama_btn.click(
            fn=refresh_ollama_models,
            outputs=[ollama_model, ollama_status]
        )

        # Test connection
        test_connection_btn.click(
            fn=test_api_connection,
            inputs=[provider, openrouter_api_key, ollama_url, openrouter_model, ollama_model],
            outputs=[connection_status]
        )

        # Generate Q&A with selected chunks
        generate_sft_btn.click(
            fn=generate_qa_from_chunks,
            inputs=[
                chunks_dataframe,
                provider,
                openrouter_api_key,
                ollama_url,
                openrouter_model,
                ollama_model,
                num_samples,
                system_prompt,
                user_prompt_template,
                temperature,
                qa_style
            ],
            outputs=[generation_output]
        )

        # Preview
        preview_btn.click(
            fn=preview_sft_data,
            outputs=[preview_output]
        )

    return {
        "generation_output": generation_output,
        "preview_output": preview_output
    }
