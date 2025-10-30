"""
Settings Tab UI
Global configuration for API keys, prompts, and defaults.
"""

import gradio as gr
from utils.config import get_global_config


def create_settings_tab():
    """
    Create settings tab for global configuration.

    Returns:
        Gradio components for the tab
    """

    config = get_global_config()

    def load_current_settings():
        """Load current settings into UI."""
        api_config = config.get_api_config()
        prompts = config.get_sft_prompts()
        training = config.get_training_defaults()
        export = config.get_export_defaults()

        return (
            api_config["openrouter_api_key"],
            api_config["ollama_base_url"],
            prompts["system_prompt"],
            prompts["user_prompt_template"],
            training["learning_rate"],
            training["epochs"],
            training["batch_size"],
            training["lora_rank"],
            training["use_4bit"],
            training["gradient_checkpointing"],
            export["format"],
            export["quantization"],
            config.get("force_cpu", False),
            config.get("max_ram_mb", 0)
        )

    def save_api_settings(api_key: str, ollama_url: str) -> str:
        """Save API configuration."""
        config.set("openrouter_api_key", api_key.strip())
        config.set("ollama_base_url", ollama_url.strip())

        if config.save():
            return " API settings saved successfully"
        else:
            return "L Failed to save API settings"

    def save_prompt_settings(system_prompt: str, user_prompt: str) -> str:
        """Save prompt configuration."""
        config.set("sft_system_prompt", system_prompt)
        config.set("sft_user_prompt_template", user_prompt)

        if config.save():
            return " Prompt settings saved successfully"
        else:
            return "L Failed to save prompt settings"

    def reset_prompts() -> tuple:
        """Reset prompts to defaults."""
        defaults = config.reset_prompts()
        config.save()
        return (
            defaults["system_prompt"],
            defaults["user_prompt_template"],
            " Prompts reset to defaults"
        )

    def save_training_defaults(lr: float, epochs: int, batch: int, rank: int, use_4bit: bool, grad_check: bool) -> str:
        """Save training defaults."""
        config.update(
            default_learning_rate=lr,
            default_epochs=int(epochs),
            default_batch_size=int(batch),
            default_lora_rank=int(rank),
            default_use_4bit=use_4bit,
            default_gradient_checkpointing=grad_check
        )

        if config.save():
            return " Training defaults saved successfully"
        else:
            return "L Failed to save training defaults"

    def reset_training_defaults() -> tuple:
        """Reset training defaults."""
        defaults = config.reset_training_defaults()
        config.save()
        return (
            defaults["default_learning_rate"],
            defaults["default_epochs"],
            defaults["default_batch_size"],
            defaults["default_lora_rank"],
            defaults["default_use_4bit"],
            defaults["default_gradient_checkpointing"],
            " Training defaults reset"
        )

    def save_export_defaults(format: str, quant: str) -> str:
        """Save export defaults."""
        config.set("default_export_format", format)
        config.set("default_quantization", quant)

        if config.save():
            return " Export defaults saved successfully"
        else:
            return "L Failed to save export defaults"

    def save_hardware_settings(force_cpu: bool, max_ram: int) -> str:
        """Save hardware override settings."""
        config.set("force_cpu", force_cpu)
        config.set("max_ram_mb", int(max_ram))

        if config.save():
            return " Hardware settings saved successfully"
        else:
            return "L Failed to save hardware settings"

    def reset_all_settings() -> str:
        """Reset all settings to defaults."""
        config.reset_to_defaults()
        if config.save():
            return " All settings reset to defaults! Reload the page to see changes."
        else:
            return "L Failed to reset settings"

    # Build UI
    with gr.Column():
        gr.Markdown("""
        # ™ Settings

        Configure global application settings. These apply to all projects.
        """)

        # API Configuration
        with gr.Accordion("= API Configuration", open=True):
            gr.Markdown("""
            Configure API keys and endpoints for external services.

            **OpenRouter API Key**: Get your key at [openrouter.ai](https://openrouter.ai/keys)
            **Ollama URL**: Local Ollama server URL (default: http://localhost:11434)
            """)

            openrouter_key = gr.Textbox(
                label="OpenRouter API Key",
                placeholder="sk-or-v1-...",
                type="password",
                info="Used for SFT data generation with GPT-4, Claude, etc."
            )

            ollama_url = gr.Textbox(
                label="Ollama Base URL",
                placeholder="http://localhost:11434",
                info="Local Ollama server endpoint"
            )

            with gr.Row():
                save_api_btn = gr.Button("Save API Settings", variant="primary")
                api_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            save_api_btn.click(
                fn=save_api_settings,
                inputs=[openrouter_key, ollama_url],
                outputs=[api_status]
            )

        # Prompt Templates
        with gr.Accordion("=Ý SFT Prompt Templates", open=False):
            gr.Markdown("""
            Customize the prompts used for SFT data generation.

            **System Prompt**: Sets the AI's role and behavior
            **User Prompt Template**: Template for generating Q&A pairs (use `{topic}` placeholder)
            """)

            system_prompt = gr.Textbox(
                label="System Prompt",
                lines=3,
                placeholder="You are an expert educational content creator...",
                info="Defines the AI's role"
            )

            user_prompt_template = gr.Textbox(
                label="User Prompt Template",
                lines=5,
                placeholder="Create a question and answer about: {topic}...",
                info="Template for generating content (use {topic} as placeholder)"
            )

            with gr.Row():
                save_prompts_btn = gr.Button("Save Prompts", variant="primary")
                reset_prompts_btn = gr.Button("Reset to Defaults", variant="secondary")

            prompts_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            save_prompts_btn.click(
                fn=save_prompt_settings,
                inputs=[system_prompt, user_prompt_template],
                outputs=[prompts_status]
            )

            reset_prompts_btn.click(
                fn=reset_prompts,
                outputs=[system_prompt, user_prompt_template, prompts_status]
            )

        # Training Defaults
        with gr.Accordion("<¯ Training Defaults", open=False):
            gr.Markdown("""
            Set default values for training parameters.
            These will be pre-filled when you open the Training tab.
            """)

            with gr.Row():
                default_lr = gr.Number(
                    label="Default Learning Rate",
                    value=2e-4,
                    info="Typical range: 1e-5 to 1e-3"
                )
                default_epochs = gr.Slider(
                    label="Default Epochs",
                    minimum=1,
                    maximum=20,
                    value=3,
                    step=1
                )

            with gr.Row():
                default_batch = gr.Slider(
                    label="Default Batch Size",
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1
                )
                default_rank = gr.Slider(
                    label="Default LoRA Rank",
                    minimum=8,
                    maximum=64,
                    value=16,
                    step=8
                )

            with gr.Row():
                default_4bit = gr.Checkbox(
                    label="4-bit Quantization by default",
                    value=True
                )
                default_grad_check = gr.Checkbox(
                    label="Gradient Checkpointing by default",
                    value=True
                )

            with gr.Row():
                save_training_btn = gr.Button("Save Training Defaults", variant="primary")
                reset_training_btn = gr.Button("Reset to Defaults", variant="secondary")

            training_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            save_training_btn.click(
                fn=save_training_defaults,
                inputs=[default_lr, default_epochs, default_batch, default_rank, default_4bit, default_grad_check],
                outputs=[training_status]
            )

            reset_training_btn.click(
                fn=reset_training_defaults,
                outputs=[default_lr, default_epochs, default_batch, default_rank, default_4bit, default_grad_check, training_status]
            )

        # Export Defaults
        with gr.Accordion("=ä Export Defaults", open=False):
            gr.Markdown("""
            Set default export format and quantization method.
            """)

            with gr.Row():
                default_format = gr.Radio(
                    label="Default Export Format",
                    choices=["GGUF", "Safetensors", "Ollama Import"],
                    value="GGUF"
                )
                default_quant = gr.Dropdown(
                    label="Default Quantization",
                    choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                    value="Q4_K_M"
                )

            save_export_btn = gr.Button("Save Export Defaults", variant="primary")
            export_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            save_export_btn.click(
                fn=save_export_defaults,
                inputs=[default_format, default_quant],
                outputs=[export_status]
            )

        # Hardware Override
        with gr.Accordion("=» Hardware Override", open=False):
            gr.Markdown("""
            **  Advanced Settings**

            Force specific hardware configurations for testing or compatibility.
            """)

            force_cpu_check = gr.Checkbox(
                label="Force CPU-only mode",
                value=False,
                info="Disable GPU acceleration (for testing)"
            )

            max_ram_input = gr.Number(
                label="Max RAM Usage (MB)",
                value=0,
                info="0 = no limit"
            )

            save_hardware_btn = gr.Button("Save Hardware Settings", variant="primary")
            hardware_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            save_hardware_btn.click(
                fn=save_hardware_settings,
                inputs=[force_cpu_check, max_ram_input],
                outputs=[hardware_status]
            )

        # Reset All
        gr.Markdown("---")
        with gr.Accordion("= Reset All Settings", open=False):
            gr.Markdown("""
            **  Danger Zone**

            This will reset ALL settings to factory defaults.
            """)

            reset_all_btn = gr.Button("Reset All Settings to Defaults", variant="stop")
            reset_all_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            reset_all_btn.click(
                fn=reset_all_settings,
                outputs=[reset_all_status]
            )

        # Load current settings on tab load
        gr.Markdown("---")
        gr.Markdown("**Current Settings Status**: Settings are loaded from `settings.json`")

        load_btn = gr.Button("= Load Current Settings", variant="secondary")
        load_btn.click(
            fn=load_current_settings,
            outputs=[
                openrouter_key, ollama_url,
                system_prompt, user_prompt_template,
                default_lr, default_epochs, default_batch, default_rank, default_4bit, default_grad_check,
                default_format, default_quant,
                force_cpu_check, max_ram_input
            ]
        )

    return {
        "openrouter_key": openrouter_key,
        "ollama_url": ollama_url,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template
    }
