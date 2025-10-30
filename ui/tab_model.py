"""
Model Selection Tab UI
Handles base model selection from Hugging Face for training.
"""

import gradio as gr
import requests
from typing import List, Tuple
from core.project_manager import ProjectManager


def create_model_tab(project_manager: ProjectManager):
    """
    Create model selection tab for Hugging Face models.

    Args:
        project_manager: ProjectManager instance

    Returns:
        Gradio components for the tab
    """

    def get_recommended_models() -> List[dict]:
        """Get recommended Unsloth-compatible models for education."""
        return [
            {
                "model_id": "unsloth/Qwen2-1.5B-Instruct",
                "display_name": "Qwen 2 (1.5B)",
                "category": "Elementary School (Grades 1-4)",
                "description": "Very fast, excellent for basic Q&A and simple explanations",
                "size": "~1GB",
                "best_for": ["Basic facts", "Simple vocabulary", "Elementary math", "Reading comprehension"],
                "hardware": "Runs on 4GB RAM, CPU-compatible"
            },
            {
                "model_id": "unsloth/Phi-3-mini-4k-instruct",
                "display_name": "Phi-3 Mini (3.8B)",
                "category": "Middle School (Grades 5-8)",
                "description": "Microsoft's compact model, exceptional at math and reasoning",
                "size": "~2.3GB",
                "best_for": ["Mathematics", "Science", "Problem solving", "STEM subjects"],
                "hardware": "Runs on 8GB RAM, GPU recommended"
            },
            {
                "model_id": "unsloth/llama-3-8b-Instruct-bnb-4bit",
                "display_name": "Llama 3 (8B)",
                "category": "High School (Grades 9-12)",
                "description": "Meta's flagship model, balanced and versatile",
                "size": "~4.7GB",
                "best_for": ["General knowledge", "Essay writing", "Complex reasoning", "Multi-subject"],
                "hardware": "Requires 12GB+ RAM, GPU recommended"
            },
            {
                "model_id": "unsloth/gemma-2-9b-it-bnb-4bit",
                "display_name": "Gemma 2 (9B)",
                "category": "High School (Grades 9-12)",
                "description": "Google's instruction-tuned model, strong factual knowledge",
                "size": "~5.4GB",
                "best_for": ["History", "Science", "Literature", "Factual accuracy"],
                "hardware": "Requires 12GB+ RAM, GPU recommended"
            },
            {
                "model_id": "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
                "display_name": "Mistral 7B (Instruct)",
                "category": "High School (Grades 9-12)",
                "description": "Efficient and powerful, good for general-purpose education",
                "size": "~4.1GB",
                "best_for": ["General education", "Language learning", "Creative writing"],
                "hardware": "Requires 10GB+ RAM, GPU recommended"
            },
            {
                "model_id": "unsloth/CodeQwen1.5-7B-Chat",
                "display_name": "CodeQwen (7B)",
                "category": "Programming Classes",
                "description": "Specialized for programming and code explanation",
                "size": "~4.5GB",
                "best_for": ["Python", "Web development", "Algorithm explanations", "Debugging"],
                "hardware": "Requires 10GB+ RAM, GPU recommended"
            }
        ]

    def validate_hf_model(model_id: str) -> Tuple[bool, str]:
        """
        Validate if Hugging Face model exists.

        Args:
            model_id: Hugging Face model ID

        Returns:
            Tuple of (is_valid, message)
        """
        if not model_id or not model_id.strip():
            return False, "Model ID cannot be empty"

        try:
            # Check if model exists on Hugging Face
            url = f"https://huggingface.co/api/models/{model_id}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return True, f"✅ Model '{model_id}' found on Hugging Face"
            elif response.status_code == 404:
                return False, f"❌ Model '{model_id}' not found on Hugging Face"
            else:
                return False, f"⚠️ Could not validate model (status {response.status_code})"

        except requests.exceptions.Timeout:
            return False, "⚠️ Validation timeout - check your internet connection"
        except Exception as e:
            return False, f"⚠️ Validation error: {str(e)}"

    def select_recommended_model(choice: str) -> Tuple[str, str]:
        """
        Auto-fill model ID from recommended selection.

        Args:
            choice: Selected recommended model

        Returns:
            Tuple of (model_id, info_message)
        """
        if not choice:
            return "", ""

        recommended = get_recommended_models()
        selected = next((m for m in recommended if m['display_name'] in choice), None)

        if selected:
            info = f"""
**{selected['display_name']}**

📚 **Category:** {selected['category']}
📝 **Description:** {selected['description']}
📦 **Size:** {selected['size']}
💻 **Hardware:** {selected['hardware']}

**Best for:**
{chr(10).join('• ' + item for item in selected['best_for'])}

**Model ID:** `{selected['model_id']}`
"""
            return selected['model_id'], info
        else:
            return "", ""

    def save_model_config(model_id: str, output_format: str) -> str:
        """
        Save model configuration to project.

        Args:
            model_id: Hugging Face model ID
            output_format: Output format choice

        Returns:
            Status message
        """
        project = project_manager.get_current_project()

        if not project:
            return "❌ Error: No project selected. Please create or select a project first."

        if not model_id or not model_id.strip():
            return "❌ Error: Please enter a Hugging Face model ID"

        # Validate model
        is_valid, validation_msg = validate_hf_model(model_id.strip())

        if not is_valid:
            return f"{validation_msg}\n\nPlease check the model ID and try again."

        # Map display format to internal format
        format_map = {
            "GGUF (Ollama/llama.cpp)": "gguf",
            "Safetensors (HuggingFace)": "safetensors",
            "Ollama Import (Auto)": "ollama"
        }

        internal_format = format_map.get(output_format, "gguf")

        # Update project config
        project.update_config({
            "base_model": model_id.strip(),
            "output_format": internal_format
        })

        return f"""✅ Configuration saved!

**Base Model:** {model_id.strip()}
**Output Format:** {output_format}

{validation_msg}

You can now proceed to:
• **Tab 3:** Generate SFT training data
• **Tab 4:** Start training
"""

    def get_current_config() -> str:
        """Get current project configuration."""
        project = project_manager.get_current_project()

        if not project:
            return "No project selected"

        base_model = project.config.get("base_model", "Not set")
        output_format = project.config.get("output_format", "Not set")

        # Format output_format for display
        format_display_map = {
            "gguf": "GGUF (Ollama/llama.cpp)",
            "safetensors": "Safetensors (HuggingFace)",
            "ollama": "Ollama Import (Auto)"
        }

        display_format = format_display_map.get(output_format, output_format)

        return f"""
**Current Configuration:**
- **Base Model:** {base_model}
- **Output Format:** {display_format}
"""

    def show_recommended_models_markdown() -> str:
        """Generate markdown for recommended models."""
        recommended = get_recommended_models()

        lines = ["## 📚 Recommended Unsloth Models for Education\n"]
        lines.append("These models are optimized for training and compatible with Unsloth.\n")

        for model in recommended:
            lines.append(f"### {model['display_name']}")
            lines.append(f"**Category:** {model['category']}")
            lines.append(f"**Model ID:** `{model['model_id']}`")
            lines.append(f"**Size:** {model['size']}")
            lines.append(f"**Hardware:** {model['hardware']}")
            lines.append(f"\n{model['description']}\n")
            lines.append("**Best for:** " + ", ".join(model['best_for']))
            lines.append("\n---\n")

        return "\n".join(lines)

    # Build UI
    with gr.Column():
        gr.Markdown("""
        ## 🎯 Base Model Selection

        Select a **Hugging Face model** to fine-tune. Use Unsloth-compatible models for best performance.

        **Note:** This is the model that will be trained. For generating SFT training data, you can use Ollama or OpenRouter in Tab 3.
        """)

        # Model ID input
        gr.Markdown("### Hugging Face Model ID")

        model_id_input = gr.Textbox(
            label="Model ID",
            placeholder="e.g., unsloth/llama-3-8b-Instruct-bnb-4bit",
            info="Enter a Hugging Face model ID (preferably Unsloth-optimized)",
            lines=1
        )

        with gr.Row():
            validate_btn = gr.Button("🔍 Validate Model", variant="secondary", scale=1)
            validation_output = gr.Textbox(label="Validation Result", interactive=False, lines=1, scale=2)

        # Recommended models
        gr.Markdown("---")
        with gr.Accordion("📚 Recommended Models", open=True):
            recommended_info = gr.Markdown(show_recommended_models_markdown())

            gr.Markdown("### Quick Select")
            recommended_dropdown = gr.Dropdown(
                label="Choose a Recommended Model",
                choices=[
                    "Qwen 2 (1.5B) - Elementary School",
                    "Phi-3 Mini (3.8B) - Middle School",
                    "Llama 3 (8B) - High School",
                    "Gemma 2 (9B) - High School",
                    "Mistral 7B - High School",
                    "CodeQwen (7B) - Programming"
                ],
                value=None,
                info="Selecting a model will auto-fill the Model ID above"
            )
            model_info_display = gr.Markdown("")

        # Output format selection
        gr.Markdown("---")
        gr.Markdown("### Output Format")

        output_format = gr.Radio(
            label="Export Format after Training",
            choices=[
                "GGUF (Ollama/llama.cpp)",
                "Safetensors (HuggingFace)",
                "Ollama Import (Auto)"
            ],
            value="GGUF (Ollama/llama.cpp)",
            info="Choose how you want to export the trained model"
        )

        gr.Markdown("""
        **Format Descriptions:**
        - **GGUF**: Optimized format for Ollama and llama.cpp. Best for local inference.
        - **Safetensors**: Standard HuggingFace format. Compatible with most ML frameworks.
        - **Ollama Import**: Exports as GGUF and automatically imports into Ollama.
        """)

        # Save configuration
        gr.Markdown("---")
        save_config_btn = gr.Button("💾 Save Configuration", variant="primary", size="lg")
        config_output = gr.Textbox(label="Status", interactive=False, lines=6)

        # Current configuration display
        with gr.Accordion("📋 Current Project Configuration", open=False):
            current_config_display = gr.Markdown(get_current_config())
            refresh_config_btn = gr.Button("🔄 Refresh")

        # Wire up events
        validate_btn.click(
            fn=lambda model_id: validate_hf_model(model_id)[1],
            inputs=[model_id_input],
            outputs=[validation_output]
        )

        recommended_dropdown.change(
            fn=select_recommended_model,
            inputs=[recommended_dropdown],
            outputs=[model_id_input, model_info_display]
        )

        save_config_btn.click(
            fn=save_model_config,
            inputs=[model_id_input, output_format],
            outputs=[config_output]
        )

        refresh_config_btn.click(
            fn=get_current_config,
            outputs=[current_config_display]
        )

    return {
        "model_id_input": model_id_input,
        "current_config_display": current_config_display
    }
