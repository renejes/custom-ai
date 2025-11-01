"""
Model training module with Apple Silicon (MPS) support.
Handles fine-tuning with LoRA/PEFT and checkpoint management.
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, List
from datetime import datetime
from datasets import Dataset


class ModelTrainer:
    """Handles model training with transformers + PEFT."""

    def __init__(self, project_path: Path):
        """Initialize trainer."""
        self.project_path = Path(project_path)
        self.checkpoints_path = self.project_path / "models" / "checkpoints"
        self.final_model_path = self.project_path / "models" / "final"
        self.logs_path = self.project_path / "logs"

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
        """Load base model for MPS."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
            elif torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
            else:
                device = "cpu"
                torch_dtype = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=None,
            )
            if device in ["mps", "cuda", "cpu"]:
                self.model = self.model.to(device)

            msg = f"Model loaded on {device}"
            if device == "mps" and load_in_4bit:
                msg += " (4-bit not available on MPS, using FP16)"

            return True, msg

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def prepare_model_for_training(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list] = None,
        use_gradient_checkpointing: bool = True
    ) -> tuple[bool, str]:
        """Add LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model

            if self.model is None:
                return False, "Model not loaded"

            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules or ["q_proj", "v_proj"],
            )

            self.model = get_peft_model(self.model, config)  # type: ignore

            if use_gradient_checkpointing:
                if hasattr(self.model, 'enable_input_require_grads'):
                    self.model.enable_input_require_grads()
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()

            return True, "LoRA adapters added"

        except Exception as e:
            return False, f"Failed to prepare model: {str(e)}"

    def load_sft_dataset(self, sft_data_path: Path) -> tuple[bool, Optional[Dataset], str]:
        """Load SFT data from JSONL."""
        try:
            if not sft_data_path.exists():
                return False, None, f"File not found: {sft_data_path}"

            samples = []
            with open(sft_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            if not samples:
                return False, None, "No training samples found"

            formatted = [
                {"instruction": s.get("instruction", ""), "output": s.get("output", "")}
                for s in samples
            ]

            dataset = Dataset.from_list(formatted)
            return True, dataset, f"Loaded {len(samples)} samples"

        except Exception as e:
            return False, None, f"Failed to load dataset: {str(e)}"

    def format_prompts(self, examples: Dict) -> Dict:
        """Format examples."""
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"""### Instruction:
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
        optim: str = "adamw_torch",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        fp16: bool = False,
        bf16: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> tuple[bool, str]:
        """Train the model."""
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

            if self.model is None or self.tokenizer is None:
                return False, "Model not loaded"

            # Load dataset
            success, dataset, msg = self.load_sft_dataset(sft_data_path)
            if not success or dataset is None:
                return False, msg

            # Tokenize dataset
            def tokenize_function(examples):
                formatted = self.format_prompts(examples)
                return self.tokenizer(formatted["text"], truncation=True, max_length=512)  # type: ignore

            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps if max_steps > 0 else -1,
                num_train_epochs=float(num_train_epochs) if max_steps <= 0 else 1.0,
                learning_rate=learning_rate,
                fp16=fp16 and not torch.backends.mps.is_available(),
                logging_steps=logging_steps,
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=3,
                report_to="none",
                use_mps_device=torch.backends.mps.is_available(),
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Train
            self.is_training = True
            self.trainer.train()
            self.is_training = False

            return True, "Training completed"

        except Exception as e:
            self.is_training = False
            return False, f"Training failed: {str(e)}"

    def save_final_model(self, output_path: Path) -> tuple[bool, str]:
        """Save final trained model."""
        try:
            if self.model is None or self.tokenizer is None:
                return False, "No model to save"

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

            return True, f"Model saved to {output_path}"

        except Exception as e:
            return False, f"Failed to save model: {str(e)}"

    def stop_training(self):
        """Stop training gracefully."""
        self.should_stop = True

    def train_cpt(
        self,
        cpt_data_path: Path,
        output_dir: Path,
        learning_rate: float = 3e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        max_seq_length: int = 2048,
        gradient_checkpointing: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> tuple[bool, str]:
        """
        Train model with CPT (Continued Pre-Training) on raw text data.

        Args:
            cpt_data_path: Path to text file with raw chunks
            output_dir: Directory to save CPT model
            learning_rate: Learning rate (higher for CPT than SFT)
            num_train_epochs: Number of epochs
            per_device_train_batch_size: Batch size
            max_seq_length: Maximum sequence length
            gradient_checkpointing: Use gradient checkpointing
            progress_callback: Optional callback for progress

        Returns:
            Tuple of (success, message)
        """
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from datasets import Dataset

            if self.model is None or self.tokenizer is None:
                return False, "Model not loaded"

            # Load raw text data
            if not cpt_data_path.exists():
                return False, f"CPT data file not found: {cpt_data_path}"

            with open(cpt_data_path, 'r', encoding='utf-8') as f:
                text_chunks = [line.strip() for line in f if line.strip()]

            if not text_chunks:
                return False, "No text chunks found in CPT data file"

            # Create dataset from raw text
            dataset = Dataset.from_dict({"text": text_chunks})

            # Tokenize dataset for language modeling
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                    return_special_tokens_mask=True
                )

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing CPT data"
            )

            # Training arguments optimized for CPT
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=per_device_train_batch_size,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                use_mps_device=torch.backends.mps.is_available(),
                gradient_checkpointing=gradient_checkpointing,
                report_to="none",
                seed=42,
                dataloader_num_workers=0,
            )

            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Train
            self.is_training = True
            self.trainer.train()
            self.is_training = False

            # Save final model
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))

            return True, f"CPT training completed. Model saved to {output_dir}"

        except Exception as e:
            self.is_training = False
            return False, f"CPT training failed: {str(e)}"

    def get_checkpoint_info(self) -> list[Dict]:
        """Get checkpoint information."""
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
