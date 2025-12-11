"""
Retrieval Module
Financial Document Intelligence Platform

This module handles:
- Embedding generation using sentence transformers
- Vector store operations with Qdrant
- Hybrid search combining semantic and keyword search
"""

from .embedder import FinancialEmbedder
from .vector_store import VectorStore, SearchResult
from .hybrid_search import HybridRetriever

__all__ = [
    'FinancialEmbedder',
    'VectorStore',
    'SearchResult',
    'HybridRetriever'
]
