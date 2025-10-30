"""
Model export module.
Handles exporting trained models to different formats (GGUF, Safetensors, Ollama).
"""

import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


class ModelExporter:
    """Exports trained models to various formats."""

    def __init__(self, project_path: Path):
        """
        Initialize exporter.

        Args:
            project_path: Path to project directory
        """
        self.project_path = Path(project_path)
        self.exports_path = self.project_path / "exports"
        self.exports_path.mkdir(parents=True, exist_ok=True)

    def export_to_gguf(
        self,
        model_path: Path,
        output_name: str,
        quantization: str = "Q4_K_M"
    ) -> Tuple[bool, str]:
        """
        Export model to GGUF format.

        Args:
            model_path: Path to trained model directory
            output_name: Output filename (without extension)
            quantization: Quantization method (Q4_K_M, Q5_K_M, Q8_0, etc.)

        Returns:
            Tuple of (success, message/path)
        """
        try:
            from unsloth import FastLanguageModel

            # Load model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )

            # Output path
            output_path = self.exports_path / f"{output_name}.gguf"

            # Export to GGUF
            model.save_pretrained_gguf(
                str(self.exports_path),
                tokenizer,
                quantization_method=quantization
            )

            # Unsloth may save with a different name, find the .gguf file
            gguf_files = list(self.exports_path.glob("*.gguf"))
            if gguf_files:
                # Rename to desired output name
                latest_gguf = max(gguf_files, key=lambda p: p.stat().st_mtime)
                if latest_gguf.name != output_path.name:
                    latest_gguf.rename(output_path)

                return True, str(output_path)
            else:
                return False, "L GGUF file not found after export"

        except ImportError:
            return False, "L Unsloth not installed"
        except Exception as e:
            return False, f"L GGUF export failed: {str(e)}"

    def export_to_safetensors(
        self,
        model_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Export model to Safetensors format (HuggingFace standard).

        Args:
            model_path: Path to trained model directory
            output_dir: Output directory (None for auto)

        Returns:
            Tuple of (success, message/path)
        """
        try:
            from unsloth import FastLanguageModel

            # Load model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )

            # Output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.exports_path / f"safetensors_{timestamp}"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as Safetensors
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            return True, str(output_dir)

        except ImportError:
            return False, "L Unsloth not installed"
        except Exception as e:
            return False, f"L Safetensors export failed: {str(e)}"

    def export_to_ollama(
        self,
        model_path: Path,
        model_name: str,
        quantization: str = "Q4_K_M"
    ) -> Tuple[bool, str]:
        """
        Export model to GGUF and import into Ollama.

        Args:
            model_path: Path to trained model directory
            model_name: Name for Ollama model (e.g., "my-tutor:latest")
            quantization: Quantization method

        Returns:
            Tuple of (success, message)
        """
        try:
            # Step 1: Export to GGUF
            gguf_name = model_name.replace(":", "-").replace("/", "-")
            success, gguf_path = self.export_to_gguf(model_path, gguf_name, quantization)

            if not success:
                return False, f"Failed to export GGUF: {gguf_path}"

            # Step 2: Create Modelfile
            modelfile_path = self.exports_path / f"{gguf_name}.Modelfile"

            modelfile_content = f"""# Modelfile for {model_name}
FROM {gguf_path}

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

# Set system prompt
SYSTEM You are a helpful educational assistant.
"""

            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Step 3: Import into Ollama
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                return True, f" Model imported to Ollama as '{model_name}'"
            else:
                return False, f"L Ollama import failed: {result.stderr}"

        except FileNotFoundError:
            return False, "L Ollama not installed or not in PATH"
        except subprocess.TimeoutExpired:
            return False, "L Ollama import timed out"
        except Exception as e:
            return False, f"L Ollama export failed: {str(e)}"

    def merge_lora_weights(
        self,
        base_model_id: str,
        lora_adapter_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Merge LoRA adapter weights into base model.

        Args:
            base_model_id: Hugging Face base model ID
            lora_adapter_path: Path to LoRA adapter
            output_dir: Output directory (None for auto)

        Returns:
            Tuple of (success, message/path)
        """
        try:
            from unsloth import FastLanguageModel

            # Load model with LoRA adapter
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(lora_adapter_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )

            # Output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.exports_path / f"merged_{timestamp}"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Merge and save
            model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")

            return True, str(output_dir)

        except ImportError:
            return False, "L Unsloth not installed"
        except Exception as e:
            return False, f"L Merge failed: {str(e)}"

    def list_exports(self) -> list[dict]:
        """
        List all exported models.

        Returns:
            List of export info dictionaries
        """
        exports = []

        if not self.exports_path.exists():
            return exports

        # List GGUF files
        for gguf_file in self.exports_path.glob("*.gguf"):
            exports.append({
                "name": gguf_file.name,
                "type": "GGUF",
                "path": str(gguf_file),
                "size": self._format_size(gguf_file.stat().st_size),
                "modified": datetime.fromtimestamp(gguf_file.stat().st_mtime).isoformat()
            })

        # List Safetensors directories
        for export_dir in self.exports_path.iterdir():
            if export_dir.is_dir() and (export_dir / "config.json").exists():
                exports.append({
                    "name": export_dir.name,
                    "type": "Safetensors",
                    "path": str(export_dir),
                    "size": self._get_dir_size(export_dir),
                    "modified": datetime.fromtimestamp(export_dir.stat().st_mtime).isoformat()
                })

        return sorted(exports, key=lambda x: x["modified"], reverse=True)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in bytes to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def _get_dir_size(directory: Path) -> str:
        """Get total size of directory."""
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return ModelExporter._format_size(total_size)


if __name__ == "__main__":
    # Test exporter
    from pathlib import Path

    project_path = Path("projects/test-project")
    exporter = ModelExporter(project_path)

    print("Exporter initialized")
    print(f"Exports path: {exporter.exports_path}")

    exports = exporter.list_exports()
    print(f"\nFound {len(exports)} exports:")
    for exp in exports:
        print(f"  - {exp['name']} ({exp['type']}) - {exp['size']}")
