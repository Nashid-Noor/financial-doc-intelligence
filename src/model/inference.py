"""
Model Inference Module
Financial Document Intelligence Platform

Handles model loading and inference for financial Q&A,
including streaming generation and batch processing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass

import torch
from loguru import logger

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer
    )
    from peft import PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. Using mock inference.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.chunker import Chunk


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = False
    repetition_penalty: float = 1.1
    num_beams: int = 1


class FinancialQAModel:
    """
    Model for financial document question answering.
    
    Supports:
    - Loading fine-tuned models with LoRA adapters
    - Standard and streaming generation
    - Batch inference
    - Context-aware prompting
    """
    
    SYSTEM_PROMPT = """You are a financial analyst AI assistant specialized in analyzing SEC filings (10-K and 10-Q reports).

Your task is to answer questions about financial documents accurately. Follow these guidelines:
- Be precise with numerical values
- Always cite the source document, page, and section when available
- If calculation is required, show your reasoning step by step
- If information is not available in the context, say so clearly
- Use professional financial terminology

Answer the question based on the provided context."""

    def __init__(self,
                 model_path: str,
                 base_model: str = None,
                 device: str = None,
                 load_in_4bit: bool = True):
        """
        Initialize the model.
        
        Args:
            model_path: Path to fine-tuned model (LoRA weights)
            base_model: Base model name (if different from saved config)
            device: Device to use ('cuda', 'cpu', or None for auto)
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        if not HAS_TRANSFORMERS:
            logger.warning("Using mock model for inference")
            self._loaded = True
            return
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Determine base model
        base_model = self.base_model
        if base_model is None:
            config_path = self.model_path / "training_config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
                    base_model = config.get("model_name")
        
        if base_model is None:
            raise ValueError("Could not determine base model. Please specify base_model.")
        
        # Configure quantization
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if it exists
        adapter_config = self.model_path / "adapter_config.json"
        if adapter_config.exists():
            logger.info("Loading LoRA adapter")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path)
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")
    
    def generate_answer(self,
                        question: str,
                        context_chunks: List[Chunk],
                        config: GenerationConfig = None) -> str:
        """
        Generate answer for a question given context chunks.
        
        Args:
            question: The question to answer
            context_chunks: List of relevant context chunks
            config: Generation configuration
            
        Returns:
            Generated answer string
        """
        if not self._loaded:
            self.load()
        
        config = config or GenerationConfig()
        
        # Build prompt
        prompt = self.build_prompt(question, context_chunks)
        
        if not HAS_TRANSFORMERS:
            return self._mock_generate(question, context_chunks)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - config.max_new_tokens
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                top_k=config.top_k if config.do_sample else None,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                num_beams=config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_stream(self,
                        question: str,
                        context_chunks: List[Chunk],
                        config: GenerationConfig = None) -> Generator[str, None, None]:
        """
        Generate answer with streaming output.
        
        Args:
            question: The question to answer
            context_chunks: List of relevant context chunks
            config: Generation configuration
            
        Yields:
            Generated text tokens
        """
        if not self._loaded:
            self.load()
        
        config = config or GenerationConfig()
        prompt = self.build_prompt(question, context_chunks)
        
        if not HAS_TRANSFORMERS:
            yield self._mock_generate(question, context_chunks)
            return
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - config.max_new_tokens
        ).to(self.model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generate in thread
        import threading
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else None,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
    
    def build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        """
        Construct prompt with context and question.
        
        Args:
            question: The question to answer
            chunks: List of context chunks
            
        Returns:
            Formatted prompt string
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = self._format_source_info(chunk)
            context_parts.append(f"[Source {i}] {source_info}\n{chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        # Apply chat template if available
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback format
        return f"""<|system|>
{self.SYSTEM_PROMPT}
<|user|>
Context:
{context}

Question: {question}
<|assistant|>
"""
    
    def _format_source_info(self, chunk: Chunk) -> str:
        """Format source information from chunk metadata."""
        parts = []
        
        if 'company' in chunk.metadata:
            parts.append(chunk.metadata['company'])
        if 'filing_type' in chunk.metadata:
            parts.append(chunk.metadata['filing_type'])
        if 'fiscal_year' in chunk.metadata:
            parts.append(str(chunk.metadata['fiscal_year']))
        if 'page_number' in chunk.metadata:
            parts.append(f"Page {chunk.metadata['page_number']}")
        if 'section_name' in chunk.metadata:
            parts.append(chunk.metadata['section_name'])
        
        return ", ".join(parts) if parts else "Unknown source"
    
    def _mock_generate(self, question: str, chunks: List[Chunk]) -> str:
        """Generate mock response for testing."""
        # Extract some info from chunks for a realistic mock response
        if chunks:
            first_chunk = chunks[0]
            company = first_chunk.metadata.get('company', 'the company')
            
            return (
                f"Based on the provided context about {company}, "
                f"I can see relevant financial information. "
                f"However, this is a mock response since the model is not loaded. "
                f"The actual model would provide a detailed answer with citations."
            )
        
        return "I don't have enough context to answer this question accurately."
    
    def batch_generate(self,
                       questions: List[str],
                       contexts: List[List[Chunk]],
                       config: GenerationConfig = None) -> List[str]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            contexts: List of context chunk lists (one per question)
            config: Generation configuration
            
        Returns:
            List of generated answers
        """
        answers = []
        for question, context in zip(questions, contexts):
            answer = self.generate_answer(question, context, config)
            answers.append(answer)
        return answers


class ModelManager:
    """
    Manages multiple models for different use cases.
    """
    
    def __init__(self):
        self.models: Dict[str, FinancialQAModel] = {}
    
    def load_model(self, name: str, model_path: str, **kwargs) -> FinancialQAModel:
        """Load and register a model."""
        model = FinancialQAModel(model_path, **kwargs)
        model.load()
        self.models[name] = model
        return model
    
    def get_model(self, name: str) -> Optional[FinancialQAModel]:
        """Get a loaded model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())


if __name__ == "__main__":
    from data_processing.chunker import Chunk, ChunkType
    
    # Example usage with mock model
    model = FinancialQAModel(
        model_path="./models/test",
        base_model="meta-llama/Llama-3.1-8B-Instruct"
    )
    
    # Sample chunks
    chunks = [
        Chunk(
            text="Apple Inc. reported total net sales of $394.3 billion for fiscal year 2022, compared to $365.8 billion in fiscal 2021, an increase of 8%.",
            chunk_type=ChunkType.TEXT,
            metadata={
                'company': 'AAPL',
                'filing_type': '10-K',
                'fiscal_year': 2022,
                'page_number': 28,
                'section_name': 'Financial Highlights'
            }
        ),
        Chunk(
            text="The increase in net sales was driven primarily by higher sales of iPhone, Services, and Mac, partially offset by lower sales of iPad.",
            chunk_type=ChunkType.TEXT,
            metadata={
                'company': 'AAPL',
                'filing_type': '10-K',
                'fiscal_year': 2022,
                'page_number': 29
            }
        )
    ]
    
    # Generate answer
    question = "What was Apple's revenue in fiscal 2022 and what drove the growth?"
    
    print("Question:", question)
    print("\nContext chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk.text[:100]}...")
    
    print("\nBuilding prompt...")
    prompt = model.build_prompt(question, chunks)
    print("\nPrompt preview:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n\nNote: To generate actual answers, load a fine-tuned model with GPU support.")
