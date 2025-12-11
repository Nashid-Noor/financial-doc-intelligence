"""
Document Chunker Module
Financial Document Intelligence Platform

Implements smart chunking that preserves semantic meaning,
handles tables separately, and maintains context overlap.
"""

import re
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np
from loguru import logger

from .pdf_parser import ParsedDocument, TableData, TextSection


class ChunkType(Enum):
    """Types of document chunks."""
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class Chunk:
    """
    Data class representing a document chunk.
    
    Contains the text content, metadata, and optional embedding.
    """
    text: str
    chunk_type: ChunkType
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None:
            self.chunk_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on content hash."""
        content = f"{self.text}:{self.metadata}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))
    
    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation: 4 chars per token)."""
        return len(self.text) // 4
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'chunk_type': self.chunk_type.value,
            'metadata': self.metadata,
            'token_count': self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Chunk':
        """Create Chunk from dictionary."""
        return cls(
            text=data['text'],
            chunk_type=ChunkType(data['chunk_type']),
            metadata=data.get('metadata', {}),
            chunk_id=data.get('chunk_id')
        )


class FinancialChunker:
    """
    Smart document chunker for financial documents.
    
    Features:
    - Preserves sentence boundaries
    - Never splits tables
    - Maintains section headers with content
    - Adds overlap for context
    - Handles financial-specific formatting
    """
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    # Patterns that shouldn't be split (e.g., numbered items, financial figures)
    NO_SPLIT_PATTERNS = [
        r'\$[\d,]+\.?\d*',  # Dollar amounts
        r'\d+\.\d+%',       # Percentages
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s+\d+',  # Dates
    ]
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 min_chunk_size: int = 100,
                 preserve_tables: bool = True):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to create
            preserve_tables: Whether to keep tables as single chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_tables = preserve_tables
        
        # Convert token sizes to character estimates (4 chars per token)
        self.chunk_chars = chunk_size * 4
        self.overlap_chars = overlap * 4
        self.min_chunk_chars = min_chunk_size * 4
    
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Chunk a parsed document with smart splitting.
        
        Args:
            document: ParsedDocument from PDF parser
            
        Returns:
            List of Chunk objects with metadata
        """
        chunks = []
        
        # Base metadata for all chunks from this document
        base_metadata = {
            'source_file': document.file_path,
            'company': document.metadata.get('company_name', 'Unknown'),
            'filing_type': document.metadata.get('filing_type', 'Unknown'),
            'fiscal_year': document.metadata.get('fiscal_year'),
            'total_pages': document.total_pages
        }
        
        # Process text sections
        text_chunks = self._chunk_text_sections(document.text_sections, base_metadata)
        chunks.extend(text_chunks)
        
        # Process tables
        if self.preserve_tables:
            table_chunks = self._process_tables(document.tables, base_metadata)
            chunks.extend(table_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_text_sections(self, 
                            sections: List[TextSection],
                            base_metadata: Dict) -> List[Chunk]:
        """Chunk text sections while preserving semantic meaning."""
        chunks = []
        
        for section in sections:
            section_metadata = {
                **base_metadata,
                'page_number': section.page_number,
                'section_name': section.section_name or 'general'
            }
            
            # Split into sentences
            sentences = self._split_into_sentences(section.text)
            
            # Group sentences into chunks
            section_chunks = self._group_sentences_into_chunks(
                sentences, section_metadata
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling financial notation."""
        # Clean the text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Split by sentence endings
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_sentences_into_chunks(self,
                                     sentences: List[str],
                                     metadata: Dict) -> List[Chunk]:
        """Group sentences into appropriately sized chunks."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_chars and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk_sentences)
                
                if len(chunk_text) >= self.min_chunk_chars:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_type=ChunkType.TEXT,
                        metadata={
                            **metadata,
                            'chunk_index': len(chunks)
                        }
                    )
                    chunks.append(chunk)
                
                # Keep overlap sentences for next chunk
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences
                )
                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_chunk_sentences)
            
            current_chunk_sentences.append(sentence)
            current_length += sentence_length
        
        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            if len(chunk_text) >= self.min_chunk_chars or not chunks:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        **metadata,
                        'chunk_index': len(chunks)
                    }
                )
                chunks.append(chunk)
            elif chunks:
                # Append to last chunk if too small
                last_chunk = chunks[-1]
                last_chunk.text += ' ' + chunk_text
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap with next chunk."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_length = 0
        
        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if overlap_length >= self.overlap_chars:
                break
            overlap_sentences.insert(0, sentence)
            overlap_length += len(sentence)
        
        return overlap_sentences
    
    def _process_tables(self, 
                       tables: List[TableData],
                       base_metadata: Dict) -> List[Chunk]:
        """Process tables as individual chunks."""
        chunks = []
        
        for table in tables:
            # Convert table to text representation
            table_text = self._table_to_text(table)
            
            # Create metadata for table
            table_metadata = {
                **base_metadata,
                'page_number': table.page_number,
                'table_index': table.table_index,
                'is_table': True,
                'num_rows': len(table.data),
                'num_cols': len(table.data.columns)
            }
            
            if table.caption:
                table_metadata['table_caption'] = table.caption
            
            chunk = Chunk(
                text=table_text,
                chunk_type=ChunkType.TABLE,
                metadata=table_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _table_to_text(self, table: TableData) -> str:
        """Convert table to searchable text representation."""
        lines = []
        
        # Add caption if available
        if table.caption:
            lines.append(f"Table: {table.caption}")
        
        # Add markdown representation
        lines.append(table.to_markdown())
        
        # Also add a plain text summary for better search
        lines.append("\nTable data summary:")
        
        # Add column headers
        columns = list(table.data.columns)
        if columns:
            lines.append(f"Columns: {', '.join(str(c) for c in columns)}")
        
        return '\n'.join(lines)
    
    def chunk_text(self, 
                   text: str,
                   metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chunk plain text without document structure.
        
        Args:
            text: Raw text to chunk
            metadata: Optional metadata to attach
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        sentences = self._split_into_sentences(text)
        return self._group_sentences_into_chunks(sentences, metadata)
    
    def chunk_with_tables(self, 
                          text: str,
                          tables: List[TableData],
                          metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chunk text that contains references to tables.
        
        Args:
            text: Document text
            tables: List of TableData objects
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects with tables as separate chunks
        """
        chunks = []
        metadata = metadata or {}
        
        # Chunk the text
        text_chunks = self.chunk_text(text, metadata)
        chunks.extend(text_chunks)
        
        # Add tables as separate chunks
        base_metadata = {**metadata}
        table_chunks = self._process_tables(tables, base_metadata)
        chunks.extend(table_chunks)
        
        return chunks


class AdaptiveChunker(FinancialChunker):
    """
    Adaptive chunker that adjusts chunk size based on content density.
    
    Uses smaller chunks for dense numerical content and larger chunks
    for narrative text.
    """
    
    def __init__(self,
                 base_chunk_size: int = 512,
                 min_chunk_size: int = 256,
                 max_chunk_size: int = 1024,
                 **kwargs):
        """
        Initialize adaptive chunker.
        
        Args:
            base_chunk_size: Default chunk size
            min_chunk_size: Minimum chunk size for dense content
            max_chunk_size: Maximum chunk size for narrative content
        """
        super().__init__(chunk_size=base_chunk_size, **kwargs)
        self.min_chunk_size_adaptive = min_chunk_size
        self.max_chunk_size_adaptive = max_chunk_size
    
    def _estimate_density(self, text: str) -> float:
        """
        Estimate information density of text.
        
        Returns value between 0 (narrative) and 1 (dense numerical).
        """
        # Count numerical content
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        dollar_amounts = re.findall(r'\$[\d,]+\.?\d*', text)
        percentages = re.findall(r'\d+\.?\d*%', text)
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Calculate density score
        numerical_chars = sum(len(n) for n in numbers)
        special_chars = sum(len(d) for d in dollar_amounts + percentages)
        
        density = (numerical_chars + special_chars * 2) / total_chars
        return min(density, 1.0)
    
    def _get_adaptive_chunk_size(self, text: str) -> int:
        """Get chunk size based on content density."""
        density = self._estimate_density(text)
        
        # Higher density = smaller chunks
        size_range = self.max_chunk_size_adaptive - self.min_chunk_size_adaptive
        adjusted_size = self.max_chunk_size_adaptive - (density * size_range)
        
        return int(adjusted_size)


def chunk_document(document: ParsedDocument,
                   chunk_size: int = 512,
                   overlap: int = 50) -> List[Chunk]:
    """Convenience function to chunk a document."""
    chunker = FinancialChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_document(document)


if __name__ == "__main__":
    # Example usage
    from .pdf_parser import PDFParser
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # Parse the document
        parser = PDFParser()
        doc = parser.parse_filing(pdf_path)
        
        # Chunk the document
        chunker = FinancialChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk_document(doc)
        
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} ({chunk.chunk_type.value}):")
            print(f"  Length: {len(chunk.text)} chars")
            print(f"  Preview: {chunk.text[:200]}...")
    else:
        print("Usage: python chunker.py <pdf_file>")
