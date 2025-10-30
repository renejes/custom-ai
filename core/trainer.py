"""
Model training module using Unsloth.
Handles fine-tuning with LoRA/QLoRA and checkpoint management.
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Callable
from datetime import datetime
from datasets import Dataset


class ModelTrainer:
    """Handles model training with Unsloth."""

    def __init__(self, project_path: Path):
        """
        Initialize trainer.

        Args:
            project_path: Path to project directory
        """
        self.project_path = Path(project_path)
        self.checkpoints_path = self.project_path / "models" / "checkpoints"
        self.final_model_path = self.project_path / "models" / "final"
        self.logs_path = self.project_path / "logs"

        # Ensure directories exist
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.final_model_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_training = False
        self.should_stop = False

    def load_model(
        self,
        model_id: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False
    ) -> tuple[bool, str]:
        """
        Load base model with Unsloth.

        Args:
            model_id: Hugging Face model ID
            max_seq_length: Maximum sequence length
            dtype: Data type (None for auto)
            load_in_4bit: Use 4-bit quantization (QLoRA)

        Returns:
            Tuple of (success, message)
        """
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            return True, f" Model '{model_id}' loaded successfully"

        except ImportError:
            return False, "L Unsloth not installed. Run: pip install unsloth"
        except Exception as e:
            return False, f"L Failed to load model: {str(e)}"

    def prepare_model_for_training(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list] = None,
        use_gradient_checkpointing: bool = True
    ) -> tuple[bool, str]:
        """
        Add LoRA adapters to model.

        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA (None for auto)
            use_gradient_checkpointing: Use gradient checkpointing

        Returns:
            Tuple of (success, message)
        """
        try:
            from unsloth import FastLanguageModel

            if self.model is None:
                return False, "L Model not loaded. Load model first."

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_rank,
                target_modules=target_modules or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            return True, " LoRA adapters added successfully"

        except Exception as e:
            return False, f"L Failed to prepare model: {str(e)}"

    def load_sft_dataset(self, sft_data_path: Path) -> tuple[bool, Optional[Dataset], str]:
        """
        Load SFT data from JSONL file.

        Args:
            sft_data_path: Path to JSONL file

        Returns:
            Tuple of (success, dataset, message)
        """
        try:
            if not sft_data_path.exists():
                return False, None, f"L SFT data file not found: {sft_data_path}"

            # Load JSONL
            samples = []
            with open(sft_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            if not samples:
                return False, None, "L No training samples found"

            # Format for training (instruction-output format)
            formatted_samples = []
            for sample in samples:
                formatted_samples.append({
                    "instruction": sample.get("instruction", ""),
                    "output": sample.get("output", "")
                })

            # Convert to Dataset
            dataset = Dataset.from_list(formatted_samples)

            return True, dataset, f" Loaded {len(samples)} training samples"

        except Exception as e:
            return False, None, f"L Failed to load dataset: {str(e)}"

    def format_prompts(self, examples: Dict) -> Dict:
        """
        Format examples into training prompts.

        Args:
            examples: Batch of examples

        Returns:
            Formatted batch with 'text' field
        """
        instructions = examples["instruction"]
        outputs = examples["output"]

        texts = []
        for instruction, output in zip(instructions, outputs):
            # Format: Alpaca-style prompt
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            texts.append(text)

        return {"text": texts}

    def train(
        self,
        sft_data_path: Path,
        output_dir: Path,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = -1,
        logging_steps: int = 1,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        fp16: bool = False,
        bf16: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> tuple[bool, str]:
        """
        Train the model.

        Args:
            sft_data_path: Path to SFT data JSONL
            output_dir: Output directory for checkpoints
            learning_rate: Learning rate
            num_train_epochs: Number of epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            max_steps: Maximum steps (-1 for no limit)
            logging_steps: Logging frequency
            save_steps: Checkpoint save frequency
            optim: Optimizer (adamw_8bit recommended)
            weight_decay: Weight decay
            lr_scheduler_type: Learning rate scheduler
            seed: Random seed
            fp16: Use FP16
            bf16: Use BF16
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, message)
        """
        try:
            from transformers import TrainingArguments
            from trl import SFTTrainer

            if self.model is None or self.tokenizer is None:
                return False, "L Model not loaded"

            # Load dataset
            success, dataset, msg = self.load_sft_dataset(sft_data_path)
            if not success:
                return False, msg

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=fp16,
                bf16=bf16,
                logging_steps=logging_steps,
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=3,  # Keep only last 3 checkpoints
                report_to="none",  # No external reporting
            )

            # Initialize trainer
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=2048,
                dataset_num_proc=2,
                packing=False,
                args=training_args,
                formatting_func=lambda examples: self.format_prompts(examples)["text"],
            )

            # Start training
            self.is_training = True
            self.should_stop = False

            # Log training start
            log_path = self.logs_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_path, 'w') as f:
                f.write(f"Training started at {datetime.now()}\n")
                f.write(f"Model: {self.model.config._name_or_path}\n")
                f.write(f"Dataset: {len(dataset)} samples\n")
                f.write(f"Epochs: {num_train_epochs}\n")
                f.write(f"Learning rate: {learning_rate}\n")
                f.write(f"Batch size: {per_device_train_batch_size}\n")
                f.write("-" * 60 + "\n")

            # Train
            self.trainer.train()

            self.is_training = False

            # Log training end
            with open(log_path, 'a') as f:
                f.write("-" * 60 + "\n")
                f.write(f"Training completed at {datetime.now()}\n")

            return True, " Training completed successfully"

        except Exception as e:
            self.is_training = False
            return False, f"L Training failed: {str(e)}"

    def save_final_model(self, output_path: Path) -> tuple[bool, str]:
        """
        Save final trained model.

        Args:
            output_path: Output directory

        Returns:
            Tuple of (success, message)
        """
        try:
            if self.model is None or self.tokenizer is None:
                return False, "L No model to save"

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model and tokenizer
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

            # Save training info
            info = {
                "saved_at": datetime.now().isoformat(),
                "model_type": "lora_adapter",
                "base_model": self.model.config._name_or_path if hasattr(self.model.config, '_name_or_path') else "unknown"
            }

            with open(output_path / "training_info.json", 'w') as f:
                json.dump(info, f, indent=2)

            return True, f" Model saved to {output_path}"

        except Exception as e:
            return False, f"L Failed to save model: {str(e)}"

    def stop_training(self):
        """Stop training gracefully."""
        self.should_stop = True
        if self.trainer is not None:
            self.trainer.args.max_steps = 0  # Trigger early stopping

    def get_checkpoint_info(self) -> list[Dict]:
        """
        Get information about saved checkpoints.

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        if not self.checkpoints_path.exists():
            return checkpoints

        for checkpoint_dir in sorted(self.checkpoints_path.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint-"):
                step = checkpoint_dir.name.split("-")[-1]
                checkpoints.append({
                    "name": checkpoint_dir.name,
                    "step": step,
                    "path": str(checkpoint_dir),
                    "modified": datetime.fromtimestamp(checkpoint_dir.stat().st_mtime).isoformat()
                })

        return checkpoints


if __name__ == "__main__":
    # Test trainer
    from pathlib import Path

    project_path = Path("projects/test-project")
    trainer = ModelTrainer(project_path)

    print("Trainer initialized")
    print(f"Checkpoints path: {trainer.checkpoints_path}")
    print(f"Final model path: {trainer.final_model_path}")
