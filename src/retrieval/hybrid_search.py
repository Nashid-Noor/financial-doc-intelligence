"""
Hybrid Search Module
Financial Document Intelligence Platform

Combines semantic (vector) and keyword (BM25) search
with reciprocal rank fusion for better retrieval.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

import numpy as np
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank-bm25 not installed. Keyword search disabled.")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.chunker import Chunk
from .embedder import FinancialEmbedder
from .vector_store import VectorStore, SearchResult


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined scores."""
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int = 0
    
    @property
    def scores(self) -> Dict[str, float]:
        return {
            'semantic': self.semantic_score,
            'keyword': self.keyword_score,
            'combined': self.combined_score
        }


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.
    
    Features:
    - Semantic search using dense embeddings
    - Keyword search using BM25
    - Reciprocal rank fusion for combining results
    - Optional re-ranking with cross-encoder
    - Metadata filtering support
    """
    
    def __init__(self,
                 vector_store: VectorStore,
                 embedder: FinancialEmbedder,
                 rerank_model: str = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: VectorStore instance for semantic search
            embedder: FinancialEmbedder instance for query encoding
            rerank_model: Optional cross-encoder model for re-ranking
        """
        self.vector_store = vector_store
        self.embedder = embedder
        
        # BM25 index
        self.bm25_index = None
        self.bm25_chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        
        # Re-ranker
        self.reranker = None
        if rerank_model:
            self._init_reranker(rerank_model)
    
    def _init_reranker(self, model_name: str):
        """Initialize cross-encoder for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(model_name)
            logger.info(f"Initialized re-ranker: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize re-ranker: {e}")
            self.reranker = None
    
    def build_bm25_index(self, chunks: List[Chunk]):
        """
        Build BM25 index for keyword search.
        
        Args:
            chunks: List of Chunk objects to index
        """
        if not HAS_BM25:
            logger.warning("BM25 not available - keyword search disabled")
            return
        
        logger.info(f"Building BM25 index with {len(chunks)} chunks")
        
        self.bm25_chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        logger.info("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Handles financial-specific terms and preserves important patterns.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Preserve financial terms with special characters
        # e.g., "10-K", "$100M", "year-over-year"
        financial_patterns = [
            (r'\$[\d,]+\.?\d*[bmk]?', lambda m: m.group().replace(',', '')),
            (r'\d+-[kq]', lambda m: m.group()),
            (r'\d+\.?\d*%', lambda m: m.group()),
        ]
        
        preserved_tokens = []
        for pattern, transform in financial_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                preserved_tokens.append(transform(match))
        
        # Basic tokenization
        tokens = re.findall(r'\b\w+\b', text)
        
        # Add preserved tokens
        tokens.extend(preserved_tokens)
        
        # Remove very short tokens (except numbers)
        tokens = [t for t in tokens if len(t) > 2 or t.isdigit()]
        
        return tokens
    
    def retrieve(self,
                 query: str,
                 top_k: int = 10,
                 semantic_weight: float = 0.6,
                 keyword_weight: float = 0.4,
                 filters: Optional[Dict] = None,
                 use_rerank: bool = True) -> List[HybridSearchResult]:
        """
        Hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            semantic_weight: Weight for semantic search scores
            keyword_weight: Weight for keyword search scores
            filters: Optional metadata filters
            use_rerank: Whether to apply re-ranking
            
        Returns:
            List of HybridSearchResult objects
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # Get semantic search results
        semantic_results = self._semantic_search(query, top_k * 2, filters)
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k * 2, filters)
        
        # Combine using reciprocal rank fusion
        combined_results = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Optional re-ranking
        if use_rerank and self.reranker is not None:
            combined_results = self._rerank_results(query, combined_results)
        
        # Take top_k and assign final ranks
        combined_results = combined_results[:top_k]
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    def _semantic_search(self,
                         query: str,
                         top_k: int,
                         filters: Optional[Dict]) -> List[Tuple[Chunk, float, int]]:
        """Perform semantic search using vector store."""
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            filter_conditions=filters,
            top_k=top_k
        )
        
        # Return as (chunk, score, rank) tuples
        return [(r.chunk, r.score, r.rank) for r in results]
    
    def _keyword_search(self,
                        query: str,
                        top_k: int,
                        filters: Optional[Dict]) -> List[Tuple[Chunk, float, int]]:
        """Perform keyword search using BM25."""
        if self.bm25_index is None:
            logger.warning("BM25 index not built - skipping keyword search")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get indices of top results
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        # Filter by metadata if specified
        results = []
        rank = 1
        for idx in top_indices:
            chunk = self.bm25_chunks[idx]
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((chunk, scores[idx], rank))
                rank += 1
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _matches_filters(self, chunk: Chunk, filters: Dict) -> bool:
        """Check if chunk matches filter conditions."""
        for field, value in filters.items():
            chunk_value = chunk.metadata.get(field)
            
            if isinstance(value, dict):
                # Range filter
                if 'gte' in value and chunk_value < value['gte']:
                    return False
                if 'lte' in value and chunk_value > value['lte']:
                    return False
            elif isinstance(value, list):
                if chunk_value not in value:
                    return False
            else:
                if chunk_value != value:
                    return False
        
        return True
    
    def _reciprocal_rank_fusion(self,
                                semantic_results: List[Tuple[Chunk, float, int]],
                                keyword_results: List[Tuple[Chunk, float, int]],
                                semantic_weight: float,
                                keyword_weight: float,
                                k: int = 60) -> List[HybridSearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result set
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            k: RRF constant (default 60)
            
        Returns:
            Combined and sorted results
        """
        # Track scores by chunk ID
        chunk_scores: Dict[str, Dict] = defaultdict(lambda: {
            'chunk': None,
            'semantic_score': 0.0,
            'keyword_score': 0.0,
            'semantic_rrf': 0.0,
            'keyword_rrf': 0.0
        })
        
        # Process semantic results
        for chunk, score, rank in semantic_results:
            chunk_id = chunk.chunk_id
            chunk_scores[chunk_id]['chunk'] = chunk
            chunk_scores[chunk_id]['semantic_score'] = score
            chunk_scores[chunk_id]['semantic_rrf'] = 1.0 / (k + rank)
        
        # Process keyword results
        for chunk, score, rank in keyword_results:
            chunk_id = chunk.chunk_id
            chunk_scores[chunk_id]['chunk'] = chunk
            chunk_scores[chunk_id]['keyword_score'] = score
            chunk_scores[chunk_id]['keyword_rrf'] = 1.0 / (k + rank)
        
        # Calculate combined scores
        results = []
        for chunk_id, data in chunk_scores.items():
            if data['chunk'] is None:
                continue
            
            combined_rrf = (
                semantic_weight * data['semantic_rrf'] +
                keyword_weight * data['keyword_rrf']
            )
            
            result = HybridSearchResult(
                chunk=data['chunk'],
                semantic_score=data['semantic_score'],
                keyword_score=data['keyword_score'],
                combined_score=combined_rrf
            )
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def _rerank_results(self,
                        query: str,
                        results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Re-rank results using cross-encoder."""
        if not results or self.reranker is None:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.chunk.text) for r in results]
        
        # Get cross-encoder scores
        try:
            rerank_scores = self.reranker.predict(pairs)
            
            # Update combined scores with rerank scores
            for i, score in enumerate(rerank_scores):
                # Combine original score with rerank score
                results[i].combined_score = (
                    0.3 * results[i].combined_score + 
                    0.7 * float(score)
                )
            
            # Re-sort by updated scores
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
        
        return results
    
    def get_chunks(self) -> List[Chunk]:
        """Get all indexed chunks."""
        return self.bm25_chunks.copy()
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray = None):
        """
        Add chunks to both semantic and keyword indices.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Optional pre-computed embeddings
        """
        # Add to vector store
        if embeddings is None:
            embeddings = self.embedder.encode_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Update BM25 index
        if HAS_BM25:
            self.bm25_chunks.extend(chunks)
            new_tokenized = [self._tokenize(c.text) for c in chunks]
            self.tokenized_corpus.extend(new_tokenized)
            self.bm25_index = BM25Okapi(self.tokenized_corpus)


class MultiRetriever:
    """
    Multi-stage retrieval pipeline.
    
    Implements a more sophisticated retrieval strategy with
    query expansion, filtering, and multi-step refinement.
    """
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        """Initialize with a base hybrid retriever."""
        self.retriever = hybrid_retriever
    
    def retrieve_with_expansion(self,
                                query: str,
                                top_k: int = 10,
                                expand_query: bool = True) -> List[HybridSearchResult]:
        """
        Retrieve with optional query expansion.
        
        Args:
            query: Search query
            top_k: Number of results
            expand_query: Whether to expand query with synonyms
            
        Returns:
            List of search results
        """
        queries = [query]
        
        if expand_query:
            expanded = self._expand_query(query)
            queries.extend(expanded)
        
        # Get results for each query
        all_results = []
        for q in queries:
            results = self.retriever.retrieve(q, top_k=top_k)
            all_results.extend(results)
        
        # Deduplicate and re-rank
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.chunk.chunk_id not in seen_ids:
                seen_ids.add(result.chunk.chunk_id)
                unique_results.append(result)
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return unique_results[:top_k]
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with financial synonyms and related terms."""
        expansions = []
        
        # Financial term mappings
        synonyms = {
            'revenue': ['sales', 'income', 'earnings'],
            'profit': ['earnings', 'income', 'net income'],
            'loss': ['deficit', 'negative earnings'],
            'growth': ['increase', 'rise', 'expansion'],
            'decrease': ['decline', 'drop', 'reduction'],
            'margin': ['profit margin', 'gross margin'],
            'debt': ['liabilities', 'borrowings', 'obligations'],
            'assets': ['holdings', 'resources'],
        }
        
        query_lower = query.lower()
        for term, alternatives in synonyms.items():
            if term in query_lower:
                for alt in alternatives[:2]:  # Limit expansions
                    expanded = query_lower.replace(term, alt)
                    expansions.append(expanded)
        
        return expansions


def create_hybrid_retriever(embedding_model: str = None,
                           vector_store_path: str = None,
                           rerank_model: str = None) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        embedding_model: Name of embedding model
        vector_store_path: Path for vector store
        rerank_model: Optional re-ranking model
        
    Returns:
        Configured HybridRetriever instance
    """
    embedder = FinancialEmbedder(model_name=embedding_model)
    vector_store = VectorStore(path=vector_store_path or ":memory:")
    vector_store.create_collection(vector_size=embedder.embedding_dim)
    
    return HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        rerank_model=rerank_model
    )


if __name__ == "__main__":
    # Example usage
    from data_processing.chunker import Chunk, ChunkType
    
    # Create retriever
    retriever = create_hybrid_retriever()
    
    # Sample chunks
    chunks = [
        Chunk(
            text="Apple Inc. reported total revenue of $394.3 billion for fiscal year 2022, representing a 8% increase from the prior year.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'AAPL', 'filing_type': '10-K', 'year': 2022}
        ),
        Chunk(
            text="The company's gross margin was 43.3% for fiscal 2022, compared to 41.8% for fiscal 2021.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'AAPL', 'filing_type': '10-K', 'year': 2022}
        ),
        Chunk(
            text="Microsoft reported revenue of $198.3 billion for fiscal year 2023, an increase of 7% year-over-year.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'MSFT', 'filing_type': '10-K', 'year': 2023}
        ),
        Chunk(
            text="Risk factors include competition, regulatory changes, and economic uncertainty.",
            chunk_type=ChunkType.TEXT,
            metadata={'company': 'AAPL', 'filing_type': '10-K', 'year': 2022, 'section': 'risk_factors'}
        ),
    ]
    
    # Add chunks to retriever
    retriever.add_chunks(chunks)
    
    # Search
    query = "What was Apple's revenue in 2022?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"Query: {query}\n")
    print("Results:")
    for result in results:
        print(f"\n[{result.rank}] Score: {result.combined_score:.4f}")
        print(f"    Semantic: {result.semantic_score:.4f}, Keyword: {result.keyword_score:.4f}")
        print(f"    Company: {result.chunk.metadata.get('company')}")
        print(f"    Text: {result.chunk.text[:100]}...")
