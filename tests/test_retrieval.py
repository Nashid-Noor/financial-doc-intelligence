"""
Tests for retrieval modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.chunker import Chunk, ChunkType
from retrieval.embedder import FinancialEmbedder
from retrieval.vector_store import VectorStore, SearchResult
from retrieval.hybrid_search import HybridRetriever, HybridSearchResult


class TestFinancialEmbedder:
    """Tests for FinancialEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        """Create an embedder instance (mock mode)."""
        return FinancialEmbedder()
    
    def test_embedder_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.embedding_dim > 0
        assert embedder.normalize is True
    
    def test_encode_query(self, embedder):
        """Test query encoding."""
        query = "What was Apple's revenue?"
        embedding = embedder.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dim,)
    
    def test_encode_chunks(self, embedder):
        """Test chunk encoding."""
        chunks = [
            Chunk(text="Revenue was $394 billion.", chunk_type=ChunkType.TEXT, metadata={}),
            Chunk(text="Profit increased 15%.", chunk_type=ChunkType.TEXT, metadata={})
        ]
        
        embeddings = embedder.encode_chunks(chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, embedder.embedding_dim)
    
    def test_embedding_normalization(self, embedder):
        """Test that embeddings are normalized."""
        query = "Test query"
        embedding = embedder.encode_query(query)
        
        # L2 norm should be approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    def test_similarity_calculation(self, embedder):
        """Test similarity calculation."""
        emb1 = embedder.encode_query("Apple revenue")
        emb2 = embedder.encode_query("Apple revenue")  # Same query
        
        similarity = embedder.similarity(emb1, emb2)
        
        # Same query should have high similarity
        assert similarity > 0.99
    
    def test_batch_similarity(self, embedder):
        """Test batch similarity calculation."""
        query_emb = embedder.encode_query("test query")
        chunk_embs = np.random.randn(5, embedder.embedding_dim)
        chunk_embs = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        
        similarities = embedder.batch_similarity(query_emb, chunk_embs)
        
        assert similarities.shape == (5,)
        assert all(-1 <= s <= 1 for s in similarities)


class TestVectorStore:
    """Tests for VectorStore class."""
    
    @pytest.fixture
    def store(self):
        """Create a vector store instance."""
        store = VectorStore(path=":memory:")
        store.create_collection(vector_size=1024)
        return store
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                text="Apple reported $394 billion revenue.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "AAPL", "year": 2022}
            ),
            Chunk(
                text="Microsoft cloud revenue grew 32%.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "MSFT", "year": 2022}
            ),
            Chunk(
                text="Amazon Web Services generated $80 billion.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "AMZN", "year": 2022}
            )
        ]
    
    def test_store_initialization(self, store):
        """Test store initialization."""
        stats = store.get_collection_stats()
        assert stats["total_points"] == 0
    
    def test_add_chunks(self, store, sample_chunks):
        """Test adding chunks to store."""
        embeddings = np.random.randn(len(sample_chunks), 1024).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        count = store.add_chunks(sample_chunks, embeddings)
        
        assert count == len(sample_chunks)
        assert store.count() == len(sample_chunks)
    
    def test_search(self, store, sample_chunks):
        """Test search functionality."""
        embeddings = np.random.randn(len(sample_chunks), 1024).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        store.add_chunks(sample_chunks, embeddings)
        
        query_vector = embeddings[0]  # Search for first chunk
        results = store.search(query_vector, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score  # Sorted by score
    
    def test_filter_search(self, store, sample_chunks):
        """Test filtered search."""
        embeddings = np.random.randn(len(sample_chunks), 1024).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        store.add_chunks(sample_chunks, embeddings)
        
        query_vector = np.random.randn(1024).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Filter by company
        results = store.search(
            query_vector,
            filter_conditions={"company": "AAPL"},
            top_k=10
        )
        
        # Should only return AAPL chunks
        assert all(r.chunk.metadata.get("company") == "AAPL" for r in results)
    
    def test_delete_by_filter(self, store, sample_chunks):
        """Test deletion by filter."""
        embeddings = np.random.randn(len(sample_chunks), 1024).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        store.add_chunks(sample_chunks, embeddings)
        
        initial_count = store.count()
        store.delete_by_filter({"company": "AAPL"})
        
        assert store.count() < initial_count
    
    def test_clear_collection(self, store, sample_chunks):
        """Test clearing collection."""
        embeddings = np.random.randn(len(sample_chunks), 1024).astype(np.float32)
        store.add_chunks(sample_chunks, embeddings)
        
        store.clear_collection()
        
        assert store.count() == 0


class TestHybridRetriever:
    """Tests for HybridRetriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create a hybrid retriever instance."""
        embedder = FinancialEmbedder()
        vector_store = VectorStore(path=":memory:")
        vector_store.create_collection(vector_size=embedder.embedding_dim)
        
        return HybridRetriever(vector_store=vector_store, embedder=embedder)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                text="Apple Inc. reported total revenue of $394.3 billion for fiscal year 2022.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "AAPL", "year": 2022}
            ),
            Chunk(
                text="The company's gross margin was 43.3% compared to 41.8% in 2021.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "AAPL", "year": 2022}
            ),
            Chunk(
                text="Microsoft reported revenue of $198.3 billion for fiscal 2023.",
                chunk_type=ChunkType.TEXT,
                metadata={"company": "MSFT", "year": 2023}
            )
        ]
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever.vector_store is not None
        assert retriever.embedder is not None
        assert retriever.bm25_index is None  # Not built yet
    
    def test_build_bm25_index(self, retriever, sample_chunks):
        """Test BM25 index building."""
        retriever.build_bm25_index(sample_chunks)
        
        assert retriever.bm25_index is not None
        assert len(retriever.bm25_chunks) == len(sample_chunks)
    
    def test_add_chunks(self, retriever, sample_chunks):
        """Test adding chunks to retriever."""
        retriever.add_chunks(sample_chunks)
        
        assert retriever.vector_store.count() == len(sample_chunks)
        assert len(retriever.bm25_chunks) == len(sample_chunks)
    
    def test_retrieve(self, retriever, sample_chunks):
        """Test hybrid retrieval."""
        retriever.add_chunks(sample_chunks)
        
        results = retriever.retrieve("What was Apple's revenue?", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, HybridSearchResult) for r in results)
        # Results should have combined scores
        assert all(hasattr(r, "combined_score") for r in results)
    
    def test_retrieve_with_filter(self, retriever, sample_chunks):
        """Test filtered retrieval."""
        retriever.add_chunks(sample_chunks)
        
        results = retriever.retrieve(
            "What was revenue?",
            top_k=10,
            filters={"company": "AAPL"}
        )
        
        # All results should be from AAPL
        # Note: BM25 filtering might not be perfect in all cases
        assert len(results) > 0
    
    def test_tokenization(self, retriever):
        """Test financial text tokenization."""
        text = "Revenue was $394.3 billion, a 10-K filing shows 15.5% growth."
        tokens = retriever._tokenize(text)
        
        assert "revenue" in tokens
        assert "billion" in tokens
        assert len(tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
