"""
Citation Extraction Module
Financial Document Intelligence Platform

Handles citation tracking, extraction, and formatting
to provide accurate source attribution for answers.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from loguru import logger

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.chunker import Chunk


@dataclass
class Citation:
    """
    Data class representing a citation to a source document.
    """
    chunk_id: str
    company: str
    filing_type: str
    page: Optional[int]
    section: Optional[str]
    text_snippet: str
    relevance_score: float
    fiscal_year: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'company': self.company,
            'filing_type': self.filing_type,
            'fiscal_year': self.fiscal_year,
            'page': self.page,
            'section': self.section,
            'text_snippet': self.text_snippet,
            'relevance_score': self.relevance_score
        }
    
    def format(self, style: str = 'full') -> str:
        """
        Format citation string.
        
        Args:
            style: 'full', 'short', or 'inline'
            
        Returns:
            Formatted citation string
        """
        if style == 'short':
            parts = [self.company, self.filing_type]
            if self.fiscal_year:
                parts.append(str(self.fiscal_year))
            return ', '.join(parts)
        
        elif style == 'inline':
            return f"[{self.company} {self.filing_type}]"
        
        else:  # full
            parts = [self.company, self.filing_type]
            if self.fiscal_year:
                parts.append(str(self.fiscal_year))
            if self.page:
                parts.append(f"Page {self.page}")
            if self.section:
                parts.append(self.section.replace('_', ' ').title())
            return ', '.join(parts)


@dataclass
class ConfidenceBreakdown:
    """Breakdown of confidence score components."""
    retrieval_score: float
    similarity_score: float
    coverage_score: float
    source_count_score: float
    
    @property
    def overall(self) -> float:
        """Calculate weighted overall confidence."""
        weights = {
            'retrieval': 0.3,
            'similarity': 0.35,
            'coverage': 0.25,
            'source_count': 0.1
        }
        return (
            weights['retrieval'] * self.retrieval_score +
            weights['similarity'] * self.similarity_score +
            weights['coverage'] * self.coverage_score +
            weights['source_count'] * self.source_count_score
        )


class CitationManager:
    """
    Manages citation extraction and formatting.
    
    Features:
    - Extract citations from source chunks
    - Match answer text to source content
    - Calculate confidence scores
    - Format citations in various styles
    """
    
    def __init__(self,
                 max_citations: int = 5,
                 min_relevance: float = 0.3,
                 snippet_length: int = 150):
        """
        Initialize citation manager.
        
        Args:
            max_citations: Maximum number of citations to return
            min_relevance: Minimum relevance score to include
            snippet_length: Length of text snippets in citations
        """
        self.max_citations = max_citations
        self.min_relevance = min_relevance
        self.snippet_length = snippet_length
    
    def extract_citations(self,
                          answer: str,
                          source_chunks: List[Chunk],
                          retrieval_scores: List[float] = None) -> List[Citation]:
        """
        Identify which source chunks support the answer.
        
        Args:
            answer: Generated answer text
            source_chunks: List of context chunks used
            retrieval_scores: Optional retrieval scores for chunks
            
        Returns:
            List of Citation objects
        """
        if not source_chunks:
            return []
        
        citations = []
        
        for i, chunk in enumerate(source_chunks):
            # Calculate relevance score
            similarity = self._calculate_similarity(answer, chunk.text)
            
            retrieval_score = retrieval_scores[i] if retrieval_scores else 0.5
            
            # Combined relevance
            relevance = 0.6 * similarity + 0.4 * retrieval_score
            
            if relevance < self.min_relevance:
                continue
            
            # Extract best matching snippet
            snippet = self._extract_best_snippet(answer, chunk.text)
            
            # Create citation
            citation = Citation(
                chunk_id=chunk.chunk_id,
                company=chunk.metadata.get('company', 'Unknown'),
                filing_type=chunk.metadata.get('filing_type', 'Unknown'),
                fiscal_year=chunk.metadata.get('fiscal_year'),
                page=chunk.metadata.get('page_number'),
                section=chunk.metadata.get('section_name'),
                text_snippet=snippet,
                relevance_score=relevance
            )
            
            citations.append(citation)
        
        # Sort by relevance and take top citations
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        return citations[:self.max_citations]
    
    def _calculate_similarity(self, answer: str, source: str) -> float:
        """
        Calculate text similarity between answer and source.
        
        Uses a combination of:
        - Word overlap (Jaccard similarity)
        - Sequence matching
        - Key term matching
        """
        # Normalize texts
        answer_lower = answer.lower()
        source_lower = source.lower()
        
        # Word overlap
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        source_words = set(re.findall(r'\b\w+\b', source_lower))
        
        if not answer_words or not source_words:
            return 0.0
        
        intersection = answer_words & source_words
        union = answer_words | source_words
        jaccard = len(intersection) / len(union)
        
        # Sequence matching for phrases
        matcher = SequenceMatcher(None, answer_lower, source_lower)
        sequence_ratio = matcher.ratio()
        
        # Key term matching (numbers, proper nouns)
        key_terms = self._extract_key_terms(answer)
        key_term_matches = sum(1 for term in key_terms if term.lower() in source_lower)
        key_term_score = key_term_matches / max(len(key_terms), 1)
        
        # Weighted combination
        similarity = 0.3 * jaccard + 0.3 * sequence_ratio + 0.4 * key_term_score
        
        return min(similarity, 1.0)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms (numbers, entities) from text."""
        terms = []
        
        # Numbers and financial figures
        numbers = re.findall(r'\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?', text)
        terms.extend(numbers)
        
        # Capitalized terms (potential entities)
        capitals = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        terms.extend(capitals)
        
        return terms
    
    def _extract_best_snippet(self, answer: str, source: str) -> str:
        """Extract the most relevant snippet from source text."""
        # Find the best matching window
        answer_lower = answer.lower()
        source_lower = source.lower()
        
        # Extract key phrases from answer
        key_phrases = self._extract_key_terms(answer)
        
        best_start = 0
        best_score = 0
        
        # Sliding window to find best snippet
        window_size = self.snippet_length
        for i in range(0, max(1, len(source) - window_size), 20):
            window = source_lower[i:i + window_size]
            score = sum(1 for phrase in key_phrases if phrase.lower() in window)
            if score > best_score:
                best_score = score
                best_start = i
        
        # Extract snippet with context
        snippet_start = max(0, best_start)
        snippet_end = min(len(source), best_start + self.snippet_length)
        
        snippet = source[snippet_start:snippet_end]
        
        # Clean up snippet boundaries
        if snippet_start > 0:
            snippet = "..." + snippet.lstrip()
        if snippet_end < len(source):
            # Try to end at sentence boundary
            last_period = snippet.rfind('.')
            if last_period > len(snippet) // 2:
                snippet = snippet[:last_period + 1]
            else:
                snippet = snippet.rstrip() + "..."
        
        return snippet
    
    def format_citation(self, chunk: Chunk, style: str = 'full') -> str:
        """
        Format a citation string from a chunk.
        
        Args:
            chunk: Source chunk
            style: Citation style ('full', 'short', 'inline')
            
        Returns:
            Formatted citation string
        """
        citation = Citation(
            chunk_id=chunk.chunk_id,
            company=chunk.metadata.get('company', 'Unknown'),
            filing_type=chunk.metadata.get('filing_type', 'Unknown'),
            fiscal_year=chunk.metadata.get('fiscal_year'),
            page=chunk.metadata.get('page_number'),
            section=chunk.metadata.get('section_name'),
            text_snippet="",
            relevance_score=1.0
        )
        
        return citation.format(style)
    
    def calculate_confidence(self,
                             answer: str,
                             chunks: List[Chunk],
                             retrieval_scores: List[float] = None) -> Tuple[float, ConfidenceBreakdown]:
        """
        Calculate overall confidence score for an answer.
        
        Args:
            answer: Generated answer
            chunks: Source chunks
            retrieval_scores: Retrieval scores for chunks
            
        Returns:
            Tuple of (confidence_score, breakdown)
        """
        if not chunks:
            breakdown = ConfidenceBreakdown(0, 0, 0, 0)
            return 0.0, breakdown
        
        # Average retrieval score
        if retrieval_scores:
            avg_retrieval = np.mean(retrieval_scores[:len(chunks)])
        else:
            avg_retrieval = 0.5
        
        # Average similarity to sources
        similarities = [
            self._calculate_similarity(answer, chunk.text) 
            for chunk in chunks
        ]
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Coverage: what fraction of answer is supported
        coverage = self._calculate_coverage(answer, chunks)
        
        # Source count: more supporting sources = higher confidence
        # Normalized to [0, 1] range
        source_count_score = min(len(chunks) / 5, 1.0)
        
        breakdown = ConfidenceBreakdown(
            retrieval_score=float(avg_retrieval),
            similarity_score=float(avg_similarity),
            coverage_score=float(coverage),
            source_count_score=float(source_count_score)
        )
        
        return breakdown.overall, breakdown
    
    def _calculate_coverage(self, answer: str, chunks: List[Chunk]) -> float:
        """Calculate what fraction of the answer is supported by sources."""
        # Extract sentences from answer
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 0.0
        
        # Combine all source text
        source_text = ' '.join(chunk.text for chunk in chunks).lower()
        
        # Check each sentence for support
        supported = 0
        for sentence in sentences:
            # Extract key terms from sentence
            terms = self._extract_key_terms(sentence)
            if terms:
                # Check if key terms appear in sources
                matches = sum(1 for term in terms if term.lower() in source_text)
                if matches >= len(terms) * 0.5:  # At least 50% of terms found
                    supported += 1
            else:
                # For sentences without key terms, use word overlap
                words = set(re.findall(r'\b\w+\b', sentence.lower()))
                source_words = set(re.findall(r'\b\w+\b', source_text))
                overlap = len(words & source_words) / max(len(words), 1)
                if overlap > 0.5:
                    supported += 1
        
        return supported / len(sentences)
    
    def add_inline_citations(self,
                             answer: str,
                             citations: List[Citation]) -> str:
        """
        Add inline citation markers to answer text.
        
        Args:
            answer: Original answer text
            citations: List of citations
            
        Returns:
            Answer with inline citations added
        """
        if not citations:
            return answer
        
        # Create citation markers
        citation_map = {c.chunk_id: f"[{i+1}]" for i, c in enumerate(citations)}
        
        # For each citation, find the best place to insert marker
        sentences = re.split(r'([.!?]+)', answer)
        
        # Combine sentences with their punctuation
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            combined.append(sentences[i] + sentences[i + 1] if i + 1 < len(sentences) else sentences[i])
        if len(sentences) % 2 == 1:
            combined.append(sentences[-1])
        
        # Add citations to relevant sentences
        result_parts = []
        for sentence in combined:
            if sentence.strip():
                # Find matching citation
                for citation in citations:
                    if self._calculate_similarity(sentence, citation.text_snippet) > 0.4:
                        marker = citation_map.get(citation.chunk_id, "")
                        if marker and marker not in sentence:
                            sentence = sentence.rstrip() + marker + " "
                            break
            result_parts.append(sentence)
        
        return ''.join(result_parts)
    
    def format_references_section(self, citations: List[Citation]) -> str:
        """
        Format a references section listing all citations.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted references string
        """
        if not citations:
            return ""
        
        lines = ["", "References:"]
        for i, citation in enumerate(citations, 1):
            lines.append(f"[{i}] {citation.format('full')}")
        
        return "\n".join(lines)


def create_cited_answer(answer: str,
                        chunks: List[Chunk],
                        retrieval_scores: List[float] = None) -> Tuple[str, List[Citation], float]:
    """
    Convenience function to create an answer with citations.
    
    Args:
        answer: Generated answer
        chunks: Source chunks
        retrieval_scores: Optional retrieval scores
        
    Returns:
        Tuple of (cited_answer, citations, confidence)
    """
    manager = CitationManager()
    
    # Extract citations
    citations = manager.extract_citations(answer, chunks, retrieval_scores)
    
    # Calculate confidence
    confidence, _ = manager.calculate_confidence(answer, chunks, retrieval_scores)
    
    # Add inline citations
    cited_answer = manager.add_inline_citations(answer, citations)
    
    # Add references section
    cited_answer += manager.format_references_section(citations)
    
    return cited_answer, citations, confidence


if __name__ == "__main__":
    from data_processing.chunker import Chunk, ChunkType
    
    # Example usage
    manager = CitationManager()
    
    # Sample chunks
    chunks = [
        Chunk(
            text="Apple Inc. reported total net sales of $394.3 billion for fiscal year 2022, representing an 8% increase from $365.8 billion in fiscal 2021.",
            chunk_type=ChunkType.TEXT,
            metadata={
                'company': 'AAPL',
                'filing_type': '10-K',
                'fiscal_year': 2022,
                'page_number': 28,
                'section_name': 'financial_highlights'
            }
        ),
        Chunk(
            text="The increase in net sales was driven primarily by higher sales of iPhone, Services, and Mac, partially offset by lower sales of iPad and Wearables.",
            chunk_type=ChunkType.TEXT,
            metadata={
                'company': 'AAPL',
                'filing_type': '10-K',
                'fiscal_year': 2022,
                'page_number': 29,
                'section_name': 'mda'
            }
        )
    ]
    
    # Sample answer
    answer = "Apple's total revenue in fiscal 2022 was $394.3 billion, which represents an 8% increase from the previous year. The growth was primarily driven by higher iPhone and Services sales."
    
    # Extract citations
    citations = manager.extract_citations(answer, chunks, [0.85, 0.72])
    
    print("Answer:", answer)
    print("\nExtracted Citations:")
    for i, c in enumerate(citations, 1):
        print(f"  [{i}] {c.format('full')}")
        print(f"      Relevance: {c.relevance_score:.2f}")
        print(f"      Snippet: {c.text_snippet[:80]}...")
    
    # Calculate confidence
    confidence, breakdown = manager.calculate_confidence(answer, chunks, [0.85, 0.72])
    print(f"\nConfidence Score: {confidence:.2f}")
    print(f"  Retrieval: {breakdown.retrieval_score:.2f}")
    print(f"  Similarity: {breakdown.similarity_score:.2f}")
    print(f"  Coverage: {breakdown.coverage_score:.2f}")
    print(f"  Source Count: {breakdown.source_count_score:.2f}")
