"""
Hardware detection module for GPU, RAM, and VRAM detection.
Provides warnings and recommendations based on available hardware.
"""

import platform
import subprocess
import psutil
import torch
from typing import Dict, List, Optional


class HardwareInfo:
    """Container for hardware information."""

    def __init__(self):
        self.device_type: str = "cpu"  # cpu, cuda, mps
        self.device_name: Optional[str] = None
        self.ram_total_gb: float = 0.0
        self.ram_available_gb: float = 0.0
        self.vram_total_gb: float = 0.0
        self.vram_available_gb: float = 0.0
        self.cpu_cores: int = 0
        self.cpu_name: str = ""
        self.platform: str = platform.system()
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_available_gb": round(self.ram_available_gb, 2),
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_available_gb": round(self.vram_available_gb, 2),
            "cpu_cores": self.cpu_cores,
            "cpu_name": self.cpu_name,
            "platform": self.platform,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"  Platform: {self.platform}",
            f"  CPU: {self.cpu_name} ({self.cpu_cores} cores)",
            f" RAM: {self.ram_available_gb:.1f}GB / {self.ram_total_gb:.1f}GB available",
        ]

        if self.device_type == "cuda":
            lines.append(f"< GPU: {self.device_name}")
            lines.append(f" VRAM: {self.vram_available_gb:.1f}GB / {self.vram_total_gb:.1f}GB available")
        elif self.device_type == "mps":
            lines.append(f"<N Apple Silicon: {self.device_name}")
            lines.append(f" Unified Memory: {self.ram_total_gb:.1f}GB")
        else:
            lines.append("  No GPU detected - using CPU (training will be slow)")

        if self.warnings:
            lines.append("\n  Warnings:")
            for warning in self.warnings:
                lines.append(f"   {warning}")

        if self.recommendations:
            lines.append("\n Recommendations:")
            for rec in self.recommendations:
                lines.append(f"   {rec}")

        return "\n".join(lines)


class HardwareDetector:
    """Detects available hardware and provides training recommendations."""

    @staticmethod
    def detect() -> HardwareInfo:
        """
        Detect all available hardware.

        Returns:
            HardwareInfo object with detected hardware specs and recommendations
        """
        info = HardwareInfo()

        # Detect CPU
        info.cpu_cores = psutil.cpu_count(logical=False) or 0
        info.cpu_name = HardwareDetector._get_cpu_name()

        # Detect RAM
        ram = psutil.virtual_memory()
        info.ram_total_gb = ram.total / (1024**3)
        info.ram_available_gb = ram.available / (1024**3)

        # Detect GPU (CUDA or MPS)
        if torch.cuda.is_available():
            info.device_type = "cuda"
            info.device_name = torch.cuda.get_device_name(0)

            # Get VRAM info
            try:
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_reserved = torch.cuda.memory_reserved(0)
                vram_allocated = torch.cuda.memory_allocated(0)

                info.vram_total_gb = vram_total / (1024**3)
                info.vram_available_gb = (vram_total - vram_reserved) / (1024**3)
            except Exception as e:
                info.warnings.append(f"Could not detect VRAM: {e}")

        elif torch.backends.mps.is_available():
            info.device_type = "mps"
            info.device_name = HardwareDetector._get_apple_chip_name()
            # MPS uses unified memory (same as RAM)
            info.vram_total_gb = info.ram_total_gb
            info.vram_available_gb = info.ram_available_gb
        else:
            info.device_type = "cpu"

        # Generate warnings and recommendations
        HardwareDetector._add_warnings_and_recommendations(info)

        return info

    @staticmethod
    def _get_cpu_name() -> str:
        """Get CPU name based on platform."""
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                return result.stdout.strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return lines[1].strip()
        except Exception:
            pass

        return "Unknown CPU"

    @staticmethod
    def _get_apple_chip_name() -> str:
        """Get Apple Silicon chip name."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            cpu_name = result.stdout.strip()

            # Extract M1/M2/M3 etc.
            if "Apple" in cpu_name:
                return cpu_name

            # Fallback: try to detect from chip
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.split("\n"):
                if "Chip:" in line:
                    return line.split(":")[1].strip()
        except Exception:
            pass

        return "Apple Silicon"

    @staticmethod
    def _add_warnings_and_recommendations(info: HardwareInfo):
        """Add warnings and recommendations based on detected hardware."""

        # RAM warnings
        if info.ram_total_gb < 8:
            info.warnings.append("Less than 8GB RAM - training may fail")
            info.recommendations.append("Upgrade to at least 16GB RAM")
        elif info.ram_total_gb < 16:
            info.warnings.append("Less than 16GB RAM - use small batch sizes")
            info.recommendations.append("Use batch_size=1 and 4-bit quantization")

        # GPU warnings
        if info.device_type == "cpu":
            info.warnings.append("No GPU detected - training will be VERY slow (50-100x slower)")
            info.recommendations.append("Consider using a system with NVIDIA GPU or Apple Silicon")
            info.recommendations.append("If training on CPU, use smallest models (1B-3B) and be patient")

        elif info.device_type == "cuda":
            if info.vram_total_gb < 6:
                info.warnings.append("Less than 6GB VRAM - only 1B models will work")
                info.recommendations.append("Use 4-bit quantization (QLoRA)")
                info.recommendations.append("Use batch_size=1")
            elif info.vram_total_gb < 12:
                info.warnings.append("Less than 12GB VRAM - stick to models d3B parameters")
                info.recommendations.append("Use 4-bit quantization for 7B models")
            else:
                info.recommendations.append(f"Good VRAM! You can train up to 7B models comfortably")

        elif info.device_type == "mps":
            if info.ram_total_gb < 16:
                info.warnings.append("Less than 16GB unified memory - stick to 1B-3B models")
                info.recommendations.append("Use batch_size=1 or 2")
            elif info.ram_total_gb >= 32:
                info.recommendations.append("Excellent unified memory! 7B models should work well")
            else:
                info.recommendations.append("Decent memory - 3B-7B models should work")

        # Batch size recommendations
        if info.device_type != "cpu":
            available_mem = info.vram_available_gb if info.device_type == "cuda" else info.ram_available_gb

            if available_mem >= 20:
                info.recommendations.append("Recommended batch_size: 4-8 (for 7B models)")
            elif available_mem >= 12:
                info.recommendations.append("Recommended batch_size: 2-4 (for 3B-7B models)")
            else:
                info.recommendations.append("Recommended batch_size: 1-2 (for 1B-3B models)")


def get_recommended_settings(info: HardwareInfo) -> Dict:
    """
    Get recommended training settings based on hardware.

    Args:
        info: HardwareInfo object

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "use_4bit": False,
        "batch_size": 1,
        "gradient_checkpointing": False,
        "max_model_size": "1B"
    }

    if info.device_type == "cpu":
        settings["use_4bit"] = True
        settings["batch_size"] = 1
        settings["gradient_checkpointing"] = True
        settings["max_model_size"] = "1B"

    elif info.device_type == "cuda":
        if info.vram_total_gb >= 24:
            settings["use_4bit"] = False
            settings["batch_size"] = 4
            settings["max_model_size"] = "13B"
        elif info.vram_total_gb >= 12:
            settings["use_4bit"] = True
            settings["batch_size"] = 2
            settings["max_model_size"] = "7B"
        elif info.vram_total_gb >= 8:
            settings["use_4bit"] = True
            settings["batch_size"] = 1
            settings["max_model_size"] = "3B"
        else:
            settings["use_4bit"] = True
            settings["batch_size"] = 1
            settings["gradient_checkpointing"] = True
            settings["max_model_size"] = "1B"

    elif info.device_type == "mps":
        if info.ram_total_gb >= 32:
            settings["batch_size"] = 4
            settings["max_model_size"] = "7B"
        elif info.ram_total_gb >= 16:
            settings["batch_size"] = 2
            settings["max_model_size"] = "3B"
        else:
            settings["batch_size"] = 1
            settings["gradient_checkpointing"] = True
            settings["max_model_size"] = "1B"

    return settings


if __name__ == "__main__":
    # Test hardware detection
    print("Detecting hardware...\n")
    hw_info = HardwareDetector.detect()
    print(hw_info.get_summary())
    print("\n" + "="*50)
    print("\nRecommended Settings:")
    settings = get_recommended_settings(hw_info)
    for key, value in settings.items():
        print(f"  {key}: {value}")
