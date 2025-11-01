"""
Model export module - MPS compatible version.
Handles exporting trained models (Safetensors only for MPS).
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime


class ModelExporter:
    """Exports trained models (MPS-compatible)."""

    def __init__(self, project_path: Path):
        """Initialize exporter."""
        self.project_path = Path(project_path)
        self.exports_path = self.project_path / "exports"
        self.exports_path.mkdir(parents=True, exist_ok=True)

    def export_to_gguf(self, model_path: Path, output_name: str, quantization: str = "Q4_K_M") -> Tuple[bool, str]:
        """Export to GGUF - requires Unsloth (not available on MPS)."""
        return False, "GGUF export requires Unsloth which is not supported on Apple Silicon. Use Safetensors export instead."

    def export_to_safetensors(self, model_path: Path, output_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """Export model to Safetensors format."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.exports_path / f"safetensors_{timestamp}"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load and save model
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            return True, str(output_dir)

        except Exception as e:
            return False, f"Safetensors export failed: {str(e)}"

    def export_to_ollama(self, model_path: Path, model_name: str, quantization: str = "Q4_K_M") -> Tuple[bool, str]:
        """Export to Ollama - requires GGUF (not available on MPS)."""
        return False, "Ollama export requires GGUF conversion which is not supported on Apple Silicon. Export to Safetensors instead."

    def merge_lora_weights(self, base_model_id: str, lora_adapter_path: Path, output_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """Merge LoRA adapter weights into base model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.exports_path / f"merged_{timestamp}"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
            
            # Load with LoRA adapter
            model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
            
            # Merge
            merged_model = model.merge_and_unload()
            
            # Save
            merged_model.save_pretrained(str(output_dir))
            
            # Also save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            tokenizer.save_pretrained(str(output_dir))

            return True, str(output_dir)

        except Exception as e:
            return False, f"Merge failed: {str(e)}"

    def list_exports(self) -> list[dict]:
        """List all exported models."""
        exports = []

        if not self.exports_path.exists():
            return exports

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
        size: float = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    @staticmethod
    def _get_dir_size(directory: Path) -> str:
        """Get total size of directory."""
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return ModelExporter._format_size(total_size)
