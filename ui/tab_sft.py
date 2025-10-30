"""
SFT Data Generation Tab UI
Handles generation of supervised fine-tuning training data using OpenRouter or Ollama.
"""

import gradio as gr
import asyncio
import json
from typing import List, Dict
from core.sft_generator import SFTGenerator
from core.ollama_client import OllamaClient
from core.project_manager import ProjectManager
from utils.config import get_global_config


def create_sft_tab(project_manager: ProjectManager):
    """
    Create SFT data generation tab.

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    # Initialize clients
    ollama_client = OllamaClient()
    config = get_global_config()

    # State for generated samples
    generated_samples = gr.State([])

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

    def refresh_ollama_models() -> tuple:
        """Refresh Ollama models list."""
        success, models, error = ollama_client.list_models()

        if success:
            if not models:
                return gr.update(choices=["No models found"], value=None), "� No Ollama models installed"

            model_names = [m["name"] for m in models]
            return gr.update(choices=model_names, value=model_names[0] if model_names else None), f" Found {len(models)} model(s)"
        else:
            return gr.update(choices=["Error"], value=None), f"L {error}"

    def test_api_connection(provider, api_key, ollama_url, openrouter_model, ollama_model) -> str:
        """Test API connection."""
        if provider == "OpenRouter":
            if not api_key.strip():
                return "L Please enter an API key"

            generator = SFTGenerator(provider="openrouter", api_key=api_key)
            success, msg = asyncio.run(generator.test_connection(openrouter_model))
            return msg

        else:  # Ollama
            generator = SFTGenerator(provider="ollama", base_url=ollama_url)
            success, msg = asyncio.run(generator.test_connection(ollama_model))
            return msg

    def generate_sft_data(
        provider,
        api_key,
        ollama_url,
        openrouter_model,
        ollama_model,
        num_samples,
        system_prompt,
        user_prompt,
        temperature,
        topic,
        progress=gr.Progress()
    ) -> tuple:
        """Generate SFT training data."""
        project = project_manager.get_current_project()

        if not project:
            return [], "L Error: No project selected", gr.update(), gr.update()

        # Validate inputs
        if num_samples < 1:
            return [], "L Number of samples must be at least 1", gr.update(), gr.update()

        if not system_prompt.strip() or not user_prompt.strip():
            return [], "L System and user prompts cannot be empty", gr.update(), gr.update()

        # Initialize generator
        if provider == "OpenRouter":
            if not api_key.strip():
                return [], "L API key is required for OpenRouter", gr.update(), gr.update()

            generator = SFTGenerator(provider="openrouter", api_key=api_key)
            model = openrouter_model
        else:  # Ollama
            generator = SFTGenerator(provider="ollama", base_url=ollama_url)
            model = ollama_model

        # Progress callback
        async def progress_callback(current, total, sample):
            progress((current, total), desc=f"Generating sample {current}/{total}")

        # Generate samples
        log_lines = [f"=� Starting generation with {provider}...\n"]
        log_lines.append(f"Model: {model}")
        log_lines.append(f"Samples: {num_samples}")
        log_lines.append(f"Topic: {topic}\n")

        try:
            success, samples, error = asyncio.run(
                generator.generate_samples(
                    model=model,
                    num_samples=int(num_samples),
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt,
                    temperature=temperature,
                    topic=topic,
                    progress_callback=progress_callback
                )
            )

            if success:
                # Save to project
                output_path = str(project.get_sft_data_path())
                save_success, save_msg = generator.save_samples(samples, output_path)

                log_lines.append(f"\n Successfully generated {len(samples)} samples")
                log_lines.append(f"\n{save_msg}")

                # Update project config
                project.update_config({
                    "sft": {
                        "provider": provider.lower(),
                        "model": model,
                        "num_samples": len(samples),
                        "temperature": temperature
                    }
                })

                # Create preview table
                preview_data = [[s["instruction"], s["output"]] for s in samples[:10]]

                return samples, "\n".join(log_lines), gr.update(value=preview_data, visible=True), gr.update(visible=True)
            else:
                log_lines.append(f"\nL Generation failed: {error}")
                return [], "\n".join(log_lines), gr.update(visible=False), gr.update(visible=False)

        except Exception as e:
            log_lines.append(f"\nL Error: {str(e)}")
            return [], "\n".join(log_lines), gr.update(visible=False), gr.update(visible=False)

    def get_default_prompts() -> tuple:
        """Get default prompts from global config."""
        config = get_global_config()
        prompts = config.get_sft_prompts()
        return prompts["system_prompt"], prompts["user_prompt_template"]

    def upload_custom_data(file) -> tuple:
        """Upload custom SFT data file."""
        project = project_manager.get_current_project()

        if not project:
            return [], "L Error: No project selected", gr.update()

        if not file:
            return [], "L No file uploaded", gr.update()

        try:
            # Read JSONL file
            samples = []
            with open(file.name, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        if "instruction" in sample and "output" in sample:
                            samples.append(sample)

            if not samples:
                return [], "L No valid samples found in file", gr.update()

            # Save to project
            output_path = str(project.get_sft_data_path())
            generator = SFTGenerator(provider="ollama")
            save_success, save_msg = generator.save_samples(samples, output_path)

            preview_data = [[s["instruction"], s["output"]] for s in samples[:10]]

            return samples, f" Uploaded {len(samples)} samples\n{save_msg}", gr.update(value=preview_data, visible=True)

        except Exception as e:
            return [], f"L Error uploading file: {str(e)}", gr.update()

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## ( SFT Data Generation

        Generate supervised fine-tuning training data using a helper AI.

        **Two options:**
        1. **OpenRouter** (Cloud): Use GPT-4, Claude, or other powerful models
        2. **Ollama** (Local): Use locally installed models (free, private)
        3. **Upload**: Provide your own JSONL file
        """)

        # Provider selection
        provider_choice = gr.Radio(
            label="Choose Provider",
            choices=["OpenRouter", "Ollama", "Upload Custom Data"],
            value="Ollama",
            info="Select how you want to generate training data"
        )

        # OpenRouter section
        with gr.Group(visible=False) as openrouter_group:
            gr.Markdown("### OpenRouter Configuration")

            api_config = config.get_api_config()

            with gr.Row():
                openrouter_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="sk-or-...",
                    value=api_config["openrouter_api_key"],
                    info="Get your API key from openrouter.ai or set it in Settings tab"
                )
                openrouter_model = gr.Dropdown(
                    label="Model",
                    choices=get_openrouter_models(),
                    value="openai/gpt-4-turbo",
                    info="Select model for generation"
                )

        # Ollama section
        with gr.Group(visible=True) as ollama_group:
            gr.Markdown("### Ollama Configuration")
            with gr.Row():
                ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=api_config["ollama_base_url"],
                    info="Ollama server URL (can be changed in Settings tab)"
                )
                ollama_model = gr.Dropdown(
                    label="Model",
                    choices=["Loading..."],
                    value=None,
                    info="Select installed Ollama model"
                )
                refresh_ollama_btn = gr.Button("=", scale=0)

            ollama_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # Upload section
        with gr.Group(visible=False) as upload_group:
            gr.Markdown("### Upload Custom Data")
            gr.Markdown("""
            Upload a JSONL file with your training data. Each line should be a JSON object with:
            ```json
            {"instruction": "question or task", "output": "answer or response"}
            ```
            """)
            custom_data_file = gr.File(
                label="Upload JSONL File",
                file_types=[".jsonl", ".json"],
                file_count="single"
            )
            upload_btn = gr.Button("=� Upload Data", variant="secondary")

        # Generation settings (hidden for upload)
        with gr.Group(visible=True) as generation_group:
            gr.Markdown("---")
            gr.Markdown("### Generation Settings")

            with gr.Row():
                num_samples = gr.Number(
                    label="Number of Samples",
                    value=100,
                    minimum=1,
                    maximum=1000,
                    step=1,
                    info="How many Q&A pairs to generate"
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="Higher = more creative, Lower = more focused"
                )

            topic = gr.Textbox(
                label="Topic",
                value="general education",
                placeholder="e.g., 5th grade mathematics, world history, Python programming",
                info="What topic should the questions be about?"
            )

            # Prompts
            with gr.Accordion("<� Customize Prompts", open=False):
                system_prompt_default, user_prompt_default = get_default_prompts()

                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=system_prompt_default,
                    lines=4,
                    info="Instructions for the AI"
                )
                user_prompt = gr.Textbox(
                    label="User Prompt Template",
                    value=user_prompt_default,
                    lines=3,
                    info="Use {topic} as placeholder"
                )
                reset_prompts_btn = gr.Button("= Reset to Defaults")

            # Action buttons
            with gr.Row():
                test_btn = gr.Button(">� Test Connection", variant="secondary")
                generate_btn = gr.Button("=� Generate Data", variant="primary", size="lg")

            test_output = gr.Textbox(label="Connection Test", interactive=False, lines=2)

        # Generation log
        generation_log = gr.Textbox(
            label="Generation Log",
            lines=12,
            interactive=False,
            placeholder="Click 'Generate Data' to start..."
        )

        # Preview
        with gr.Accordion("=@ Sample Preview", open=False, visible=False) as preview_accordion:
            preview_table = gr.Dataframe(
                headers=["Instruction", "Output"],
                label="First 10 Samples",
                wrap=True,
                interactive=False
            )

        # Current data info
        with gr.Accordion("=� Current Project Data", open=False):
            def get_current_data_info():
                project = project_manager.get_current_project()
                if not project:
                    return "No project selected"

                sft_path = project.get_sft_data_path()
                if not sft_path.exists():
                    return "No SFT data generated yet"

                try:
                    with open(sft_path, 'r') as f:
                        lines = f.readlines()
                        return f"**Current SFT Data:**\n- Samples: {len(lines)}\n- Path: `{sft_path}`"
                except:
                    return "Error reading SFT data"

            current_data_display = gr.Markdown(get_current_data_info())
            refresh_data_btn = gr.Button("= Refresh")

        # Wire up events
        def toggle_groups(choice):
            """Toggle visibility based on provider choice."""
            return (
                gr.update(visible=choice == "OpenRouter"),  # openrouter_group
                gr.update(visible=choice == "Ollama"),  # ollama_group
                gr.update(visible=choice == "Upload Custom Data"),  # upload_group
                gr.update(visible=choice != "Upload Custom Data")  # generation_group
            )

        provider_choice.change(
            fn=toggle_groups,
            inputs=[provider_choice],
            outputs=[openrouter_group, ollama_group, upload_group, generation_group]
        )

        refresh_ollama_btn.click(
            fn=refresh_ollama_models,
            outputs=[ollama_model, ollama_status]
        )

        test_btn.click(
            fn=test_api_connection,
            inputs=[provider_choice, openrouter_api_key, ollama_url, openrouter_model, ollama_model],
            outputs=[test_output]
        )

        generate_btn.click(
            fn=generate_sft_data,
            inputs=[
                provider_choice, openrouter_api_key, ollama_url,
                openrouter_model, ollama_model, num_samples,
                system_prompt, user_prompt, temperature, topic
            ],
            outputs=[generated_samples, generation_log, preview_table, preview_accordion]
        )

        upload_btn.click(
            fn=upload_custom_data,
            inputs=[custom_data_file],
            outputs=[generated_samples, generation_log, preview_table]
        )

        reset_prompts_btn.click(
            fn=get_default_prompts,
            outputs=[system_prompt, user_prompt]
        )

        refresh_data_btn.click(
            fn=get_current_data_info,
            outputs=[current_data_display]
        )

    # Initial load of Ollama models
    def initial_load():
        return refresh_ollama_models()

    return {
        "generated_samples": generated_samples,
        "current_data_display": current_data_display,
        "initial_load": initial_load
    }
