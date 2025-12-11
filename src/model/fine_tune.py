"""
Fine-tuning Module
Financial Document Intelligence Platform

Fine-tunes Llama 3.1 8B using QLoRA for financial document Q&A.
Includes data preparation, training loop, and evaluation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml

import torch
from loguru import logger

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/peft not installed. Fine-tuning disabled.")


@dataclass
class TrainingConfig:
    """Configuration for model fine-tuning."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./models/finqa-llama-3.1-8b"
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 2048
    
    # Optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'output_dir': self.output_dir,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'load_in_4bit': self.load_in_4bit,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'max_length': self.max_length,
        }


class FinancialQATrainer:
    """
    Trainer for financial document Q&A model.
    
    Uses QLoRA for efficient fine-tuning on consumer hardware.
    """
    
    SYSTEM_PROMPT = """You are a financial analyst AI assistant specialized in analyzing SEC filings (10-K and 10-Q reports). 

Your task is to answer questions about financial documents accurately. Follow these guidelines:
- Be precise with numerical values
- Always cite the source document, page, and section when available
- If calculation is required, show your reasoning step by step
- If information is not available in the context, say so clearly
- Use professional financial terminology

Answer the question based on the provided context."""

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_model(self) -> Tuple[Any, Any]:
        """
        Load and prepare the model for fine-tuning.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers/peft not installed")
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self._print_trainable_params()
        
        return self.model, self.tokenizer
    
    def _print_trainable_params(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """
        Prepare dataset for training.
        
        Expected format:
        {
            "question": "What was revenue?",
            "context": "Revenue was $100B...",
            "answer": "Revenue was $100 billion. Source: 10-K page 23."
        }
        
        Args:
            data: List of training examples
            
        Returns:
            HuggingFace Dataset
        """
        formatted_examples = []
        
        for example in data:
            # Format as conversation
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._format_user_message(example)},
                {"role": "assistant", "content": example.get("answer", "")}
            ]
            
            # Apply chat template
            text = self._apply_chat_template(messages)
            formatted_examples.append({"text": text})
        
        return Dataset.from_list(formatted_examples)
    
    def _format_user_message(self, example: Dict) -> str:
        """Format user message with question and context."""
        question = example.get("question", "")
        context = example.get("context", "")
        
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}"
        else:
            return f"Question: {question}"
    
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template for the model."""
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        
        # Fallback format
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|{role}|>\n{content}\n"
        return formatted
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )
        
        return dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"]
        )
    
    def train(self,
              train_dataset: Dataset,
              eval_dataset: Dataset = None,
              callbacks: List = None) -> Dict:
        """
        Run the training loop.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of callbacks
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.prepare_model()
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.tokenize_dataset(eval_dataset)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=eval_dataset is not None,
            fp16=self.config.fp16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="none",  # Disable wandb by default
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the final model
        self.save_model()
        
        return train_result.metrics
    
    def save_model(self, path: str = None):
        """Save the fine-tuned model."""
        save_path = path or self.config.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        # Save LoRA weights
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        with open(Path(save_path) / "training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def evaluate(self, eval_dataset: Dataset) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        eval_dataset = self.tokenize_dataset(eval_dataset)
        return self.trainer.evaluate(eval_dataset)


class DataPreparer:
    """
    Prepare training data from various financial QA datasets.
    """
    
    @staticmethod
    def load_finqa(path: str) -> List[Dict]:
        """Load and format FinQA dataset."""
        with open(path) as f:
            data = json.load(f)
        
        formatted = []
        for item in data:
            question = item.get("qa", {}).get("question", "")
            answer = item.get("qa", {}).get("exe_ans", "")
            
            # Combine text and table context
            context_parts = []
            if "text" in item:
                context_parts.append(item["text"])
            if "table" in item:
                # Format table as markdown
                table = item["table"]
                if table:
                    context_parts.append(DataPreparer._format_table(table))
            
            formatted.append({
                "question": question,
                "context": "\n\n".join(context_parts),
                "answer": str(answer),
                "source": "finqa"
            })
        
        return formatted
    
    @staticmethod
    def load_tatqa(path: str) -> List[Dict]:
        """Load and format TAT-QA dataset."""
        with open(path) as f:
            data = json.load(f)
        
        formatted = []
        for item in data:
            for qa in item.get("questions", []):
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                context_parts = []
                if "paragraphs" in item:
                    for para in item["paragraphs"]:
                        context_parts.append(para.get("text", ""))
                
                if "table" in item:
                    context_parts.append(DataPreparer._format_table(item["table"]))
                
                formatted.append({
                    "question": question,
                    "context": "\n\n".join(context_parts),
                    "answer": str(answer),
                    "source": "tatqa"
                })
        
        return formatted
    
    @staticmethod
    def _format_table(table: List[List]) -> str:
        """Format table as markdown."""
        if not table:
            return ""
        
        lines = []
        for i, row in enumerate(table):
            line = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(line)
            if i == 0:
                # Add separator after header
                lines.append("|" + "|".join("---" for _ in row) + "|")
        
        return "\n".join(lines)
    
    @staticmethod
    def create_train_val_test_split(data: List[Dict],
                                    train_ratio: float = 0.8,
                                    val_ratio: float = 0.1,
                                    seed: int = 42) -> Tuple[List, List, List]:
        """Split data into train/val/test sets."""
        import random
        random.seed(seed)
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            data[:train_end],
            data[train_end:val_end],
            data[val_end:]
        )


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="./models/test-finqa",
        num_epochs=1,
        batch_size=2
    )
    
    # Sample training data
    sample_data = [
        {
            "question": "What was Apple's total revenue in fiscal 2022?",
            "context": "Apple Inc. reported total net sales of $394.3 billion for fiscal year 2022, compared to $365.8 billion in fiscal 2021.",
            "answer": "Apple's total revenue in fiscal 2022 was $394.3 billion. Source: Annual Report, Net Sales section."
        },
        {
            "question": "What was the year-over-year revenue growth?",
            "context": "Apple Inc. reported total net sales of $394.3 billion for fiscal year 2022, compared to $365.8 billion in fiscal 2021.",
            "answer": "The year-over-year revenue growth was approximately 7.8%, calculated as ($394.3B - $365.8B) / $365.8B × 100."
        }
    ]
    
    print("Training config:")
    print(json.dumps(config.to_dict(), indent=2))
    
    print("\nNote: To actually train, install transformers, peft, and bitsandbytes")
    print("Then run with real training data and GPU support.")
