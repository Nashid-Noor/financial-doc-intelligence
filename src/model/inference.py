"""
Model Inference Module
Financial Document Intelligence Platform

Handles model inference using Hugging Face Inference API.
"""

import os
from typing import List, Generator, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from huggingface_hub import InferenceClient

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = False
    repetition_penalty: float = 1.1

class FinancialQAModel:
    """Wrapper for Hugging Face Inference API."""
    
    SYSTEM_PROMPT = """You are a financial analyst AI assistant specialized in analyzing SEC filings (10-K and 10-Q reports).

Your task is to answer questions about financial documents accurately. Follow these guidelines:
- Be precise with numerical values
- Always cite the source document, page, and section when available
- If calculation is required, show your reasoning step by step
- If information is not available in the context, say so clearly
- Use professional financial terminology

Answer the question based on the provided context."""

    def __init__(self,
                 model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 api_key: str = None,
                 **kwargs):
        """
        Initialize the model client.
        
        Args:
            model_path: Hugging Face model ID
            api_key: Hugging Face API token
        """
        self.model_id = model_path
        self.api_key = api_key or os.getenv("HF_API_KEY")
        
        if not self.api_key:
            logger.warning("HF_API_KEY not found. API calls will fail unless public access is sufficient.")
            
        self.client = InferenceClient(token=self.api_key)
        logger.info(f"Initialized HF Inference Client for {self.model_id}")

    
    def generate_answer(self,
                        question: str,
                        context_chunks: List[Any],
                        config: GenerationConfig = None) -> str:
        """
        Generate answer for a question given context chunks.
        """
        config = config or GenerationConfig()
        messages = self.build_messages(question, context_chunks)
        
        try:
            response = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=42,
                stream=False
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling HF API: {e}")
            return f"Error generating answer: {str(e)}"

    def generate_stream(self,
                        question: str,
                        context_chunks: List[Any],
                        config: GenerationConfig = None) -> Generator[str, None, None]:
        """
        Generate answer with streaming output.
        """
        config = config or GenerationConfig()
        messages = self.build_messages(question, context_chunks)
        
        try:
            stream = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=42,
                stream=True
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"Error streaming from HF API: {e}")
            yield f"Error: {str(e)}"

    def build_messages(self, question: str, chunks: List[Any]) -> List[Dict[str, str]]:
        """
        Construct messages for chat API.
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = self._format_source_info(chunk)
            context_parts.append(f"[Source {i}] {source_info}\n{chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long (approx 12k chars ~ 3-4k tokens) to avoid API errors
        max_chars = 12000
        if len(context) > max_chars:
            logger.warning(f"Context too long ({len(context)} chars), truncating to {max_chars}")
            context = context[:max_chars] + "...[TRUNCATED]"
        
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

    def _format_source_info(self, chunk: Any) -> str:
        """Format source information from chunk metadata."""
        parts = []
        metadata = chunk.metadata
        
        if 'company' in metadata:
            parts.append(metadata['company'])
        if 'filing_type' in metadata:
            parts.append(metadata['filing_type'])
        if 'fiscal_year' in metadata:
            parts.append(str(metadata['fiscal_year']))
        if 'page_number' in metadata:
            parts.append(f"Page {metadata['page_number']}")
        if 'section_name' in metadata:
            parts.append(metadata['section_name'])
        
        return ", ".join(parts) if parts else "Unknown source"
