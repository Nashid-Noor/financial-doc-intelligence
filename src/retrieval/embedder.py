"""
Embedding Generation Module
Financial Document Intelligence Platform

Generates embeddings for document chunks and queries using
sentence transformers, with batching and caching support.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Union
import pickle

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using mock embedder.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.chunker import Chunk


class FinancialEmbedder:
    """
    Generate embeddings for financial documents.
    
    Uses BGE-large or similar embedding models optimized for
    retrieval tasks. Supports batch processing and caching.
    """
    
    # Default model for financial document embeddings
    DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Instruction prefix for BGE models (improves retrieval)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    def __init__(self,
                 model_name: str = None,
                 device: str = None,
                 cache_dir: Optional[str] = None,
                 normalize_embeddings: bool = True):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache embeddings
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.normalize = normalize_embeddings
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # Load the model
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        else:
            logger.warning("Using mock embedder - install sentence-transformers for real embeddings")
            self.model = None
            self.embedding_dim = 1024  # Default for BGE-large
    
    def encode_chunks(self,
                      chunks: List[Chunk],
                      batch_size: int = 32,
                      show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of shape (n_chunks, embedding_dim)
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
        
        Uses query instruction prefix for better retrieval with BGE models.
        
        Args:
            query: Query string
            
        Returns:
            NumPy array of shape (embedding_dim,)
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
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding
            
        Returns:
            NumPy array of shape (n_queries, embedding_dim)
        """
        # Add instruction for BGE models
        if "bge" in self.model_name.lower():
            queries = [self.QUERY_INSTRUCTION + q for q in queries]
        
        return self._encode_texts(queries, batch_size, show_progress=False)
    
    def _encode_texts(self,
                      texts: List[str],
                      batch_size: int,
                      show_progress: bool) -> np.ndarray:
        """Internal method to encode texts."""
        if self.model is None:
            # Mock embeddings for testing without model
            return self._mock_encode(texts)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        )
        
        return np.array(embeddings)
    
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
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if self.normalize:
            return float(np.dot(embedding1, embedding2))
        else:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def batch_similarity(self,
                         query_embedding: np.ndarray,
                         chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between a query and multiple chunk embeddings.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            chunk_embeddings: Chunk embeddings of shape (n_chunks, embedding_dim)
            
        Returns:
            Similarity scores of shape (n_chunks,)
        """
        if self.normalize:
            return np.dot(chunk_embeddings, query_embedding)
        else:
            query_norm = np.linalg.norm(query_embedding)
            chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
            
            # Avoid division by zero
            chunk_norms = np.maximum(chunk_norms, 1e-9)
            
            dot_products = np.dot(chunk_embeddings, query_embedding)
            return dot_products / (chunk_norms * query_norm)


class CachedEmbedder(FinancialEmbedder):
    """
    Embedder with persistent caching.
    
    Stores embeddings in a SQLite database for efficient
    lookup across sessions.
    """
    
    def __init__(self,
                 model_name: str = None,
                 cache_db: str = "embeddings_cache.db",
                 **kwargs):
        """
        Initialize cached embedder.
        
        Args:
            model_name: Name of the embedding model
            cache_db: Path to SQLite database for caching
            **kwargs: Additional arguments for FinancialEmbedder
        """
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
    """
    Factory function to create an embedder.
    
    Args:
        model_name: Name of the embedding model
        cache_dir: Directory for file caching
        use_db_cache: Whether to use SQLite database caching
        
    Returns:
        FinancialEmbedder or CachedEmbedder instance
    """
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
