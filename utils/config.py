"""
Global configuration management.
Handles persistent settings across application sessions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class GlobalConfig:
    """Manages global application settings."""

    DEFAULT_SETTINGS = {
        # API Configuration
        "openrouter_api_key": "",
        "ollama_base_url": "http://localhost:11434",

        # Default Prompts (SFT Generation)
        "sft_system_prompt": "You are an expert educational content creator. Generate high-quality question-answer pairs for training an AI tutor.",
        "sft_user_prompt_template": "Create a clear, educational question and answer about: {topic}\n\nFormat your response as JSON:\n{{\n  \"instruction\": \"the question\",\n  \"output\": \"the detailed answer\"\n}}",

        # Training Defaults
        "default_learning_rate": 2e-4,
        "default_epochs": 3,
        "default_batch_size": 1,
        "default_lora_rank": 16,
        "default_use_4bit": True,
        "default_gradient_checkpointing": True,

        # Hardware Override
        "force_cpu": False,
        "max_ram_mb": 0,  # 0 = no limit

        # UI Preferences
        "theme": "soft",
        "show_tooltips": True,
        "show_advanced_options": False,

        # Export Defaults
        "default_export_format": "GGUF",
        "default_quantization": "Q4_K_M",
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize global config.

        Args:
            config_path: Path to settings.json (None for default)
        """
        if config_path is None:
            self.config_path = Path("settings.json")
        else:
            self.config_path = Path(config_path)

        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)

                # Merge with defaults (in case new settings were added)
                settings = self.DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
            except Exception as e:
                print(f"  Failed to load settings: {e}")
                return self.DEFAULT_SETTINGS.copy()
        else:
            return self.DEFAULT_SETTINGS.copy()

    def save(self) -> bool:
        """
        Save current settings to file.

        Returns:
            Success status
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"L Failed to save settings: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key
            default: Default value if key doesn't exist

        Returns:
            Setting value
        """
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value.

        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value

    def update(self, **kwargs) -> None:
        """
        Update multiple settings.

        Args:
            **kwargs: Key-value pairs to update
        """
        self.settings.update(kwargs)

    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()

    def reset_prompts(self) -> Dict[str, str]:
        """
        Reset prompts to defaults.

        Returns:
            Dict with system_prompt and user_prompt_template
        """
        self.settings["sft_system_prompt"] = self.DEFAULT_SETTINGS["sft_system_prompt"]
        self.settings["sft_user_prompt_template"] = self.DEFAULT_SETTINGS["sft_user_prompt_template"]

        return {
            "system_prompt": self.settings["sft_system_prompt"],
            "user_prompt_template": self.settings["sft_user_prompt_template"]
        }

    def reset_training_defaults(self) -> Dict[str, Any]:
        """
        Reset training defaults.

        Returns:
            Dict with training parameters
        """
        defaults = {
            "default_learning_rate": self.DEFAULT_SETTINGS["default_learning_rate"],
            "default_epochs": self.DEFAULT_SETTINGS["default_epochs"],
            "default_batch_size": self.DEFAULT_SETTINGS["default_batch_size"],
            "default_lora_rank": self.DEFAULT_SETTINGS["default_lora_rank"],
            "default_use_4bit": self.DEFAULT_SETTINGS["default_use_4bit"],
            "default_gradient_checkpointing": self.DEFAULT_SETTINGS["default_gradient_checkpointing"],
        }

        self.settings.update(defaults)
        return defaults

    def get_api_config(self) -> Dict[str, str]:
        """
        Get API configuration.

        Returns:
            Dict with openrouter_api_key and ollama_base_url
        """
        return {
            "openrouter_api_key": self.settings.get("openrouter_api_key", ""),
            "ollama_base_url": self.settings.get("ollama_base_url", "http://localhost:11434")
        }

    def get_sft_prompts(self) -> Dict[str, str]:
        """
        Get SFT generation prompts.

        Returns:
            Dict with system_prompt and user_prompt_template
        """
        return {
            "system_prompt": self.settings.get("sft_system_prompt", self.DEFAULT_SETTINGS["sft_system_prompt"]),
            "user_prompt_template": self.settings.get("sft_user_prompt_template", self.DEFAULT_SETTINGS["sft_user_prompt_template"])
        }

    def get_training_defaults(self) -> Dict[str, Any]:
        """
        Get training default parameters.

        Returns:
            Dict with training parameters
        """
        return {
            "learning_rate": self.settings.get("default_learning_rate", 2e-4),
            "epochs": self.settings.get("default_epochs", 3),
            "batch_size": self.settings.get("default_batch_size", 1),
            "lora_rank": self.settings.get("default_lora_rank", 16),
            "use_4bit": self.settings.get("default_use_4bit", True),
            "gradient_checkpointing": self.settings.get("default_gradient_checkpointing", True),
        }

    def get_export_defaults(self) -> Dict[str, str]:
        """
        Get export default settings.

        Returns:
            Dict with export_format and quantization
        """
        return {
            "format": self.settings.get("default_export_format", "GGUF"),
            "quantization": self.settings.get("default_quantization", "Q4_K_M")
        }

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings.

        Returns:
            Complete settings dictionary
        """
        return self.settings.copy()


# Global instance (singleton pattern)
_global_config_instance = None


def get_global_config() -> GlobalConfig:
    """
    Get or create global config instance.

    Returns:
        GlobalConfig instance
    """
    global _global_config_instance

    if _global_config_instance is None:
        _global_config_instance = GlobalConfig()

    return _global_config_instance


if __name__ == "__main__":
    # Test config
    config = GlobalConfig("test_settings.json")

    print("Default settings:")
    print(json.dumps(config.get_all_settings(), indent=2))

    # Test get/set
    config.set("openrouter_api_key", "test-key-123")
    print(f"\nAPI Key: {config.get('openrouter_api_key')}")

    # Test save/load
    config.save()
    print("\n Settings saved to test_settings.json")
