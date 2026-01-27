"""
Embedding Generation Module
Financial Document Intelligence Platform

Generates embeddings using Hugging Face Inference API.
"""

import os
import hashlib
import time
import random
from pathlib import Path
from typing import List, Optional
import pickle
import numpy as np
from loguru import logger
from huggingface_hub import InferenceClient

from data_processing.chunker import Chunk

class FinancialEmbedder:
    """
    Generate embeddings for financial documents using HF API.
    """
    
    # Default model for financial document embeddings
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Instruction prefix for BGE models (improves retrieval)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    def __init__(self,
                 model_name: str = None,
                 cache_dir: Optional[str] = None,
                 api_key: str = None,
                 normalize_embeddings: bool = True):
        """
        Initialize the embedder.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.normalize = normalize_embeddings
        self.api_key = api_key or os.getenv("HF_API_KEY")
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        if not self.api_key:
            logger.warning("HF_API_KEY not found. Embedding generation might fail.")
            
        self.client = InferenceClient(token=self.api_key)
        self.embedding_dim = 384  # Default for MiniLM
        logger.info(f"Initialized HF Embedder for {self.model_name}")

    def encode_chunks(self,
                      chunks: List[Chunk],
                      batch_size: int = 32,
                      show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for document chunks.
        """
        if not chunks:
            return np.array([])
        
        texts = [chunk.text for chunk in chunks]
        
        # Check cache
        if self.cache_dir:
            cache_key = self._get_cache_key(texts)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded {len(texts)} embeddings from cache")
                return cached
        
        # Generate embeddings
        embeddings = self._encode_texts(texts, batch_size, show_progress)
        
        if len(embeddings) == 0 and len(chunks) > 0:
            raise ValueError("Embedding generation failed. Check API key and model status.")

        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings.")

        # Cache results
        if self.cache_dir:
            self._save_to_cache(cache_key, embeddings)
        
        # Attach embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        """
        # Add instruction for BGE models
        if "bge" in self.model_name.lower():
            query = self.QUERY_INSTRUCTION + query
        
        return self._encode_texts([query], batch_size=1, show_progress=False)[0]
    
    def encode_queries(self,
                       queries: List[str],
                       batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple queries.
        """
        # Add instruction for BGE models
        if "bge" in self.model_name.lower():
            queries = [self.QUERY_INSTRUCTION + q for q in queries]
        
        return self._encode_texts(queries, batch_size, show_progress=False)
    
    def _encode_texts(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Retry logic
            max_retries = 5
            retry_count = 0
            backoff = 2  # Start with 2 seconds
            
            while retry_count < max_retries:
                try:
                    # Use feature extraction API
                    embeddings = self.client.feature_extraction(
                        text=batch,
                        model=self.model_name
                    )
                    
                    if isinstance(embeddings, list):
                        embeddings = np.array(embeddings)
                    
                    # BGE model returns normalized embeddings usually, but we can re-normalize
                    if self.normalize:
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        embeddings = embeddings / np.maximum(norms, 1e-9)
                        
                    all_embeddings.append(embeddings)
                    break # Success
                    
                except Exception as e:
                    # Check if it's a client error (4xx) -> Fail fast
                    # Except 429 (Too Many Requests), which should be retried
                    is_client_error = False
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status = e.response.status_code
                        if 400 <= status < 500 and status != 429:
                            is_client_error = True
                    
                    if is_client_error:
                        logger.error(f"Permanent HF API Error: {e}")
                        return np.array([])
                        
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Error calling HF Embedding API after {max_retries} attempts: {e}")
                        return np.array([])
                    
                    # Exponential backoff with jitter
                    wait_time = backoff + random.uniform(0, 1)
                    logger.warning(f"Embedding API error (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    backoff *= 2
        
        if not all_embeddings:
            return np.array([])
            
        return np.vstack(all_embeddings)
    
    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for testing."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.randn(self.embedding_dim)
            if self.normalize:
                embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for a list of texts."""
        combined = "||".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, key: str, embeddings: np.ndarray):
        """Save embeddings to cache."""
        if self.cache_dir is None:
            return
        
        cache_path = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(embedding1, embedding2))
    
    def batch_similarity(self,
                         query_embedding: np.ndarray,
                         chunk_embeddings: np.ndarray) -> np.ndarray:
        """Calculate batch similarity."""
        return np.dot(chunk_embeddings, query_embedding)


class CachedEmbedder(FinancialEmbedder):
    """
    Embedder with persistent caching (SQLite).
    """
    
    def __init__(self,
                 model_name: str = None,
                 cache_db: str = "embeddings_cache.db",
                 **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.cache_db = cache_db
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize the cache database."""
        import sqlite3
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if it exists."""
        import sqlite3
        text_hash = hashlib.md5(text.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT embedding FROM embeddings_cache WHERE text_hash = ? AND model_name = ?',
            (text_hash, self.model_name)
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return pickle.loads(result[0])
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding."""
        import sqlite3
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_blob = pickle.dumps(embedding)
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings_cache (text_hash, embedding, model_name)
            VALUES (?, ?, ?)
        ''', (text_hash, embedding_blob, self.model_name))
        conn.commit()
        conn.close()


def create_embedder(model_name: str = None,
                    cache_dir: str = None,
                    use_db_cache: bool = False) -> FinancialEmbedder:
    """Factory function."""
    if use_db_cache:
        return CachedEmbedder(model_name=model_name)
    else:
        return FinancialEmbedder(model_name=model_name, cache_dir=cache_dir)


if __name__ == "__main__":
    # Example usage
    embedder = FinancialEmbedder()
    
    # Test with sample texts
    texts = [
        "Apple Inc. reported revenue of $394.3 billion for fiscal year 2022.",
        "The company's gross margin was 43.3% compared to 41.8% in 2021.",
        "Operating expenses increased 14% year-over-year."
    ]
    
    # Create mock chunks
    chunks = [Chunk(text=t, chunk_type="text", metadata={}) for t in texts]
    
    # Generate embeddings
    embeddings = embedder.encode_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings[0].shape}")
    
    # Test query encoding
    query = "What was Apple's revenue?"
    query_embedding = embedder.encode_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Calculate similarities
    similarities = embedder.batch_similarity(query_embedding, embeddings)
    print(f"Similarities: {similarities}")
