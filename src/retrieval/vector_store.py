"""
Vector Store Module
Financial Document Intelligence Platform

Provides vector storage and retrieval using Qdrant,
with support for metadata filtering and batch operations.
"""

import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, Range
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    logger.warning("qdrant-client not installed. Using mock vector store.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.chunker import Chunk


@dataclass
class SearchResult:
    """Result from a vector search."""
    chunk: Chunk
    score: float
    rank: int = 0


class VectorStore:
    """
    Vector store using Qdrant for efficient similarity search.
    
    Features:
    - Metadata filtering (company, filing type, date)
    - Batch upsert for efficiency
    - Multiple distance metrics
    - Persistent and in-memory modes
    """
    
    DEFAULT_COLLECTION = "financial_docs"
    
    def __init__(self,
                 collection_name: str = None,
                 host: str = "localhost",
                 port: int = 6333,
                 path: str = None,
                 embedding_dim: int = 1024):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host (for client mode)
            port: Qdrant server port
            path: Path for local storage (None for in-memory)
            embedding_dim: Dimension of embedding vectors
        """
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.embedding_dim = embedding_dim
        
        if HAS_QDRANT:
            if path == ":memory:" or path is None:
                logger.info("Initializing in-memory Qdrant")
                self.client = QdrantClient(":memory:")
            elif path:
                logger.info(f"Initializing local Qdrant at {path}")
                Path(path).mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=path)
            else:
                logger.info(f"Connecting to Qdrant at {host}:{port}")
                self.client = QdrantClient(host=host, port=port)
        else:
            logger.warning("Using mock vector store")
            self.client = None
            self._mock_data = {}
    
    def create_collection(self,
                          vector_size: int = None,
                          distance: str = "cosine",
                          recreate: bool = False):
        """
        Initialize vector collection with schema.
        
        Args:
            vector_size: Size of embedding vectors
            distance: Distance metric ('cosine', 'euclid', 'dot')
            recreate: Whether to recreate if collection exists
        """
        vector_size = vector_size or self.embedding_dim
        
        if self.client is None:
            self._mock_data = {}
            return
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and not recreate:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
        
        if exists and recreate:
            logger.info(f"Recreating collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)
        
        # Map distance metric
        distance_map = {
            'cosine': Distance.COSINE,
            'euclid': Distance.EUCLID,
            'dot': Distance.DOT
        }
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )
        
        logger.info(f"Created collection '{self.collection_name}' "
                   f"(dim={vector_size}, distance={distance})")
    
    def add_chunks(self,
                   chunks: List[Chunk],
                   embeddings: np.ndarray = None,
                   batch_size: int = 100) -> int:
        """
        Add document chunks to vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Pre-computed embeddings (or None to use chunk.embedding)
            batch_size: Batch size for upsert operations
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Get embeddings
        if embeddings is None:
            embeddings = np.array([c.embedding for c in chunks])
        
        if self.client is None:
            # Mock storage
            for i, chunk in enumerate(chunks):
                self._mock_data[chunk.chunk_id] = {
                    'chunk': chunk,
                    'embedding': embeddings[i]
                }
            return len(chunks)
        
        # Prepare points for Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            point_id = chunk.chunk_id or str(uuid.uuid4())
            
            # Prepare payload (metadata)
            payload = {
                'text': chunk.text,
                'chunk_type': chunk.chunk_type.value,
                **chunk.metadata
            }
            
            point = PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload
            )
            points.append(point)
        
        # Batch upsert
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_added += len(batch)
            
            if len(points) > batch_size:
                logger.debug(f"Added batch {i // batch_size + 1}, "
                           f"total: {total_added}/{len(points)}")
        
        logger.info(f"Added {total_added} chunks to vector store")
        return total_added
    
    def search(self,
               query_vector: np.ndarray,
               filter_conditions: Optional[Dict] = None,
               top_k: int = 10,
               score_threshold: float = None) -> List[SearchResult]:
        """
        Semantic search with optional filtering.
        
        Args:
            query_vector: Query embedding
            filter_conditions: Dict of metadata filters
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        if self.client is None:
            return self._mock_search(query_vector, top_k)
        
        # Build filter
        qdrant_filter = self._build_filter(filter_conditions)
        
        # Search
        # Search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        results = response.points
        
        # Convert to SearchResult objects
        search_results = []
        for rank, hit in enumerate(results):
            chunk = Chunk(
                text=hit.payload.get('text', ''),
                chunk_type=hit.payload.get('chunk_type', 'text'),
                metadata={k: v for k, v in hit.payload.items() 
                         if k not in ['text', 'chunk_type']},
                chunk_id=str(hit.id)
            )
            
            search_results.append(SearchResult(
                chunk=chunk,
                score=hit.score,
                rank=rank + 1
            ))
        
        return search_results
    
    def _build_filter(self, conditions: Optional[Dict]) -> Optional[Filter]:
        """Build Qdrant filter from conditions dict."""
        if not conditions:
            return None
        
        must_conditions = []
        
        for field, value in conditions.items():
            if isinstance(value, dict):
                # Range filter
                if 'gte' in value or 'lte' in value:
                    must_conditions.append(
                        FieldCondition(
                            key=field,
                            range=Range(
                                gte=value.get('gte'),
                                lte=value.get('lte')
                            )
                        )
                    )
            elif isinstance(value, list):
                # Match any in list
                for v in value:
                    must_conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=v)
                        )
                    )
            else:
                # Exact match
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=must_conditions) if must_conditions else None
    
    def _mock_search(self, 
                     query_vector: np.ndarray,
                     top_k: int) -> List[SearchResult]:
        """Mock search for testing without Qdrant."""
        results = []
        
        for chunk_id, data in self._mock_data.items():
            # Calculate cosine similarity
            embedding = data['embedding']
            similarity = np.dot(query_vector, embedding) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding)
            )
            
            results.append(SearchResult(
                chunk=data['chunk'],
                score=float(similarity),
                rank=0
            ))
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def delete_by_filter(self, filter_conditions: Dict) -> int:
        """
        Delete chunks matching filter conditions.
        
        Args:
            filter_conditions: Dict of metadata filters
            
        Returns:
            Number of deleted points
        """
        if self.client is None:
            # Mock deletion
            to_delete = []
            for chunk_id, data in self._mock_data.items():
                match = True
                for field, value in filter_conditions.items():
                    if data['chunk'].metadata.get(field) != value:
                        match = False
                        break
                if match:
                    to_delete.append(chunk_id)
            
            for chunk_id in to_delete:
                del self._mock_data[chunk_id]
            return len(to_delete)
        
        qdrant_filter = self._build_filter(filter_conditions)
        
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter)
        )
        
        return result.operation_id
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        if self.client is None:
            return {
                'total_points': len(self._mock_data),
                'status': 'mock'
            }
        
        info = self.client.get_collection(self.collection_name)
        
        return {
            'total_points': info.points_count,
            'indexed_points': info.indexed_vectors_count,
            'status': info.status.value,
            'vector_size': info.config.params.vectors.size
        }
    
    def clear_collection(self):
        """Delete all points in the collection."""
        if self.client is None:
            self._mock_data = {}
            return
        
        # Recreate collection to clear all data
        info = self.client.get_collection(self.collection_name)
        self.create_collection(
            vector_size=info.config.params.vectors.size,
            recreate=True
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a specific chunk by ID."""
        if self.client is None:
            data = self._mock_data.get(chunk_id)
            return data['chunk'] if data else None
        
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id]
        )
        
        if results:
            hit = results[0]
            return Chunk(
                text=hit.payload.get('text', ''),
                chunk_type=hit.payload.get('chunk_type', 'text'),
                metadata={k: v for k, v in hit.payload.items() 
                         if k not in ['text', 'chunk_type']},
                chunk_id=str(hit.id)
            )
        
        return None
    
    def count(self, filter_conditions: Optional[Dict] = None) -> int:
        """Count chunks, optionally with filter."""
        if self.client is None:
            if not filter_conditions:
                return len(self._mock_data)
            # Apply filter to mock data
            count = 0
            for data in self._mock_data.values():
                match = True
                for field, value in filter_conditions.items():
                    if data['chunk'].metadata.get(field) != value:
                        match = False
                        break
                if match:
                    count += 1
            return count
        
        qdrant_filter = self._build_filter(filter_conditions)
        
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_filter
        )
        
        return result.count


class VectorStoreManager:
    """
    Manages multiple vector store collections.
    
    Useful for organizing documents by company, filing type, etc.
    """
    
    def __init__(self, base_path: str = "qdrant_data"):
        """Initialize the manager with a base storage path."""
        self.base_path = Path(base_path)
        self.stores: Dict[str, VectorStore] = {}
    
    def get_store(self, collection_name: str) -> VectorStore:
        """Get or create a vector store for the given collection."""
        if collection_name not in self.stores:
            self.stores[collection_name] = VectorStore(
                collection_name=collection_name,
                path=str(self.base_path / collection_name)
            )
            self.stores[collection_name].create_collection()
        
        return self.stores[collection_name]
    
    def list_collections(self) -> List[str]:
        """List all managed collections."""
        return list(self.stores.keys())


if __name__ == "__main__":
    # Example usage
    from data_processing.chunker import Chunk, ChunkType
    
    # Create vector store
    store = VectorStore(path=":memory:")
    store.create_collection(vector_size=1024)
    
    # Create sample chunks
    chunks = [
        Chunk(
            text="Apple Inc. reported revenue of $394.3 billion for fiscal year 2022.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'AAPL', 'filing_type': '10-K', 'year': 2022}
        ),
        Chunk(
            text="Microsoft's cloud revenue grew 32% year-over-year.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'MSFT', 'filing_type': '10-K', 'year': 2022}
        ),
        Chunk(
            text="Amazon Web Services generated $80 billion in revenue.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'AMZN', 'filing_type': '10-K', 'year': 2022}
        )
    ]
    
    # Generate mock embeddings
    embeddings = np.random.randn(len(chunks), 1024).astype(np.float32)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Add to store
    store.add_chunks(chunks, embeddings)
    
    print(f"Added {len(chunks)} chunks")
    print(f"Stats: {store.get_collection_stats()}")
    
    # Search
    query_embedding = np.random.randn(1024).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = store.search(query_embedding, top_k=3)
    print(f"\nSearch results:")
    for result in results:
        print(f"  [{result.rank}] Score: {result.score:.3f} - {result.chunk.text[:50]}...")
