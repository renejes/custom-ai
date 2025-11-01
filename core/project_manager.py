"""
Project management module for creating, loading, and managing training projects.
Each project has its own directory with config, data, and models.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Project:
    """Represents a training project."""

    def __init__(self, name: str, base_path: str = "projects", custom_location: Optional[str] = None):
        self.name = name

        # Support custom project location
        if custom_location:
            self.project_path = Path(custom_location)
        else:
            self.base_path = Path(base_path)
            self.project_path = self.base_path / name

        self.config_path = self.project_path / "config.json"

        # Organized folder structure
        self.data_path = self.project_path / "data"
        self.databases_path = self.data_path / "databases"  # All RAG databases

        self.models_path = self.project_path / "models"
        self.cpt_model_path = self.models_path / "cpt_model"  # CPT trained model
        self.sft_model_path = self.models_path / "sft_model"  # SFT trained model
        self.checkpoints_path = self.models_path / "checkpoints"  # Training checkpoints
        self.final_models_path = self.models_path / "final"  # Final exported models

        self.logs_path = self.project_path / "logs"

        self.config: Dict = {}

    def exists(self) -> bool:
        """Check if project exists."""
        return self.project_path.exists() and self.config_path.exists()

    def create(self, description: str = "") -> bool:
        """
        Create new project with directory structure.

        Args:
            description: Optional project description

        Returns:
            True if created successfully, False if already exists
        """
        if self.exists():
            return False

        # Create organized directory structure
        self.project_path.mkdir(parents=True, exist_ok=True)

        # Data folders
        self.data_path.mkdir(exist_ok=True)
        self.databases_path.mkdir(exist_ok=True)

        # Model folders
        self.models_path.mkdir(exist_ok=True)
        self.cpt_model_path.mkdir(exist_ok=True)
        self.sft_model_path.mkdir(exist_ok=True)
        self.checkpoints_path.mkdir(exist_ok=True)
        self.final_models_path.mkdir(exist_ok=True)

        # Logs folder
        self.logs_path.mkdir(exist_ok=True)

        # Create default config
        self.config = {
            "name": self.name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "base_model": None,
            "output_format": "gguf",  # gguf, safetensors, ollama
            "training": {
                "learning_rate": 2e-4,
                "batch_size": 1,
                "epochs": 3,
                "lora_rank": 16,
                "lora_alpha": 32,
                "use_4bit": False,
                "gradient_checkpointing": False,
                "max_seq_length": 2048,
            },
            "sft": {
                "provider": "openrouter",  # openrouter or ollama
                "api_key": "",
                "model": "",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "num_samples": 100,
                "system_prompt": "",
                "user_prompt_template": "",
            },
            "rag": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "documents_processed": 0,
                "total_chunks": 0,
            }
        }

        self.save_config()
        return True

    def load(self) -> bool:
        """
        Load project configuration.

        Returns:
            True if loaded successfully, False if project doesn't exist
        """
        if not self.exists():
            return False

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        return True

    def save_config(self):
        """Save configuration to disk."""
        self.config["updated_at"] = datetime.now().isoformat()

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def update_config(self, updates: Dict):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary with configuration updates (nested)
        """
        def deep_update(d: Dict, u: Dict) -> Dict:
            """Deep update dictionary."""
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = deep_update(self.config, updates)
        self.save_config()

    def delete(self) -> bool:
        """
        Delete project and all its files.

        Returns:
            True if deleted successfully
        """
        if not self.exists():
            return False

        shutil.rmtree(self.project_path)
        return True

    def get_rag_db_path(self) -> Path:
        """Get path to default RAG database."""
        return self.databases_path / "wissensbasis.db"

    def get_sft_data_path(self) -> Path:
        """Get path to SFT training data."""
        return self.data_path / "sft_data.jsonl"

    def get_info(self) -> Dict:
        """Get project information summary."""
        info = {
            "name": self.name,
            "description": self.config.get("description", ""),
            "created_at": self.config.get("created_at", ""),
            "updated_at": self.config.get("updated_at", ""),
            "base_model": self.config.get("base_model", "Not selected"),
            "has_rag_data": self.get_rag_db_path().exists(),
            "has_sft_data": self.get_sft_data_path().exists(),
            "num_checkpoints": len(list(self.checkpoints_path.glob("*"))) if self.checkpoints_path.exists() else 0,
            "has_final_model": len(list(self.final_models_path.glob("*"))) > 0 if self.final_models_path.exists() else False,
        }

        return info


class ProjectManager:
    """Manages all projects."""

    def __init__(self, base_path: str = "projects"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.current_project: Optional[Project] = None

    def list_projects(self) -> List[str]:
        """
        List all available projects.

        Returns:
            List of project names
        """
        if not self.base_path.exists():
            return []

        projects = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                projects.append(item.name)

        return sorted(projects)

    def create_project(self, name: str, description: str = "") -> Optional[Project]:
        """
        Create a new project.

        Args:
            name: Project name (must be valid directory name)
            description: Optional project description

        Returns:
            Project object if created, None if already exists or invalid name
        """
        # Validate project name
        if not name or not self._is_valid_project_name(name):
            return None

        project = Project(name, str(self.base_path))

        if project.exists():
            return None

        if project.create(description):
            return project

        return None

    def load_project(self, name: str) -> Optional[Project]:
        """
        Load an existing project.

        Args:
            name: Project name

        Returns:
            Project object if exists, None otherwise
        """
        project = Project(name, str(self.base_path))

        if project.load():
            self.current_project = project
            return project

        return None

    def delete_project(self, name: str) -> bool:
        """
        Delete a project.

        Args:
            name: Project name

        Returns:
            True if deleted successfully
        """
        project = Project(name, str(self.base_path))

        # Unset current project if it's being deleted
        if self.current_project and self.current_project.name == name:
            self.current_project = None

        return project.delete()

    def get_current_project(self) -> Optional[Project]:
        """Get currently loaded project."""
        return self.current_project

    def set_current_project(self, name: str) -> Optional[Project]:
        """
        Set current project by name.

        Args:
            name: Project name

        Returns:
            Project object if loaded successfully
        """
        return self.load_project(name)

    @staticmethod
    def _is_valid_project_name(name: str) -> bool:
        """
        Check if project name is valid.

        Args:
            name: Project name to validate

        Returns:
            True if valid
        """
        # No empty names
        if not name.strip():
            return False

        # No special characters that would break file systems
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        if any(char in name for char in invalid_chars):
            return False

        # No leading/trailing dots or spaces
        if name.startswith('.') or name.endswith('.'):
            return False

        if name.startswith(' ') or name.endswith(' '):
            return False

        return True


if __name__ == "__main__":
    # Test project manager
    manager = ProjectManager()

    # Create test project
    print("Creating project 'test_project'...")
    project = manager.create_project("test_project", "A test project")

    if project:
        print(f" Project created at: {project.project_path}")
        print(f" Config: {project.config}")
        print(f"\nProject info: {project.get_info()}")
    else:
        print(" Failed to create project")

    # List projects
    print(f"\nAvailable projects: {manager.list_projects()}")

    # Load project
    print("\nLoading project...")
    loaded = manager.load_project("test_project")
    if loaded:
        print(f" Loaded: {loaded.name}")
        print(f"  Config: {loaded.config.get('description')}")
