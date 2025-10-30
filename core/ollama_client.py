"""
Ollama API client for listing and managing local models.
"""

import subprocess
import requests
import json
from typing import List, Dict, Optional, Tuple


class OllamaClient:
    """Client for interacting with Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"

    def is_running(self) -> bool:
        """
        Check if Ollama is running.

        Returns:
            True if Ollama is accessible
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> Tuple[bool, List[Dict], str]:
        """
        List all available models.

        Returns:
            Tuple of (success, models_list, error_message)
        """
        try:
            # Try API first
            response = requests.get(f"{self.api_url}/tags", timeout=5)

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                # Format model data
                formatted_models = []
                for model in models:
                    formatted_models.append({
                        "name": model.get("name", ""),
                        "size": self._format_size(model.get("size", 0)),
                        "modified": model.get("modified_at", ""),
                        "family": model.get("details", {}).get("family", ""),
                        "parameter_size": model.get("details", {}).get("parameter_size", ""),
                        "quantization": model.get("details", {}).get("quantization_level", "")
                    })

                return True, formatted_models, ""
            else:
                return False, [], f"API returned status {response.status_code}"

        except requests.exceptions.ConnectionError:
            # Try subprocess as fallback
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    models = self._parse_ollama_list_output(result.stdout)
                    return True, models, ""
                else:
                    return False, [], "Ollama command failed"
            except FileNotFoundError:
                return False, [], "Ollama not installed or not in PATH"
            except subprocess.TimeoutExpired:
                return False, [], "Ollama command timed out"
            except Exception as e:
                return False, [], f"Subprocess error: {str(e)}"

        except Exception as e:
            return False, [], f"Error: {str(e)}"

    def get_model_info(self, model_name: str) -> Tuple[bool, Dict, str]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (success, model_info, error_message)
        """
        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": model_name},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return True, data, ""
            else:
                return False, {}, f"Failed to get model info: {response.status_code}"

        except Exception as e:
            return False, {}, f"Error: {str(e)}"

    def pull_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Pull a model from Ollama library.

        Args:
            model_name: Name of the model to pull

        Returns:
            Tuple of (success, message)
        """
        try:
            # Use subprocess for pull (shows progress)
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            if result.returncode == 0:
                return True, f"Successfully pulled {model_name}"
            else:
                return False, f"Failed to pull model: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Pull operation timed out (10 minutes)"
        except FileNotFoundError:
            return False, "Ollama not installed or not in PATH"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Delete a model.

        Args:
            model_name: Name of the model to delete

        Returns:
            Tuple of (success, message)
        """
        try:
            response = requests.delete(
                f"{self.api_url}/delete",
                json={"name": model_name},
                timeout=30
            )

            if response.status_code == 200:
                return True, f"Successfully deleted {model_name}"
            else:
                return False, f"Failed to delete model: {response.status_code}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in bytes to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def _parse_ollama_list_output(output: str) -> List[Dict]:
        """
        Parse output from 'ollama list' command.

        Args:
            output: Raw output from ollama list

        Returns:
            List of model dictionaries
        """
        models = []
        lines = output.strip().split('\n')

        # Skip header line
        if len(lines) <= 1:
            return models

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                models.append({
                    "name": parts[0],
                    "size": parts[2] if len(parts) > 2 else "Unknown",
                    "modified": " ".join(parts[3:]) if len(parts) > 3 else "",
                    "family": "",
                    "parameter_size": "",
                    "quantization": ""
                })

        return models

    def get_recommended_models(self) -> List[Dict]:
        """
        Get list of recommended models for educational use.

        Returns:
            List of recommended model configurations
        """
        return [
            {
                "name": "qwen2:1.5b",
                "display_name": "Qwen 2 (1.5B)",
                "category": "Elementary School (Grades 1-4)",
                "description": "Very fast, good for basic Q&A and simple explanations",
                "size": "~1GB",
                "recommended_for": ["Basic facts", "Simple vocabulary", "Elementary math"]
            },
            {
                "name": "phi3:mini",
                "display_name": "Phi-3 Mini (3.8B)",
                "category": "Middle School (Grades 5-8)",
                "description": "Excellent reasoning for math and science, compact size",
                "size": "~2.3GB",
                "recommended_for": ["Mathematics", "Science", "Reading comprehension"]
            },
            {
                "name": "llama3:8b",
                "display_name": "Llama 3 (8B)",
                "category": "High School (Grades 9-12)",
                "description": "Balanced performance, good language quality",
                "size": "~4.7GB",
                "recommended_for": ["General knowledge", "Essay writing", "Complex reasoning"]
            },
            {
                "name": "gemma2:9b",
                "display_name": "Gemma 2 (9B)",
                "category": "High School (Grades 9-12)",
                "description": "Google's model, strong at factual knowledge",
                "size": "~5.4GB",
                "recommended_for": ["History", "Science", "Literature"]
            },
            {
                "name": "codegemma:7b",
                "display_name": "CodeGemma (7B)",
                "category": "Programming Classes",
                "description": "Specialized for code and programming concepts",
                "size": "~5GB",
                "recommended_for": ["Python", "Web development", "Algorithm explanations"]
            }
        ]


if __name__ == "__main__":
    # Test Ollama client
    client = OllamaClient()

    print("Checking if Ollama is running...")
    if client.is_running():
        print(" Ollama is running")

        print("\nListing models...")
        success, models, error = client.list_models()

        if success:
            print(f"Found {len(models)} models:")
            for model in models:
                print(f"  - {model['name']} ({model['size']})")
        else:
            print(f" Error: {error}")
    else:
        print(" Ollama is not running")
        print("\nRecommended models:")
        for model in client.get_recommended_models():
            print(f"  - {model['display_name']}: {model['description']}")
