"""
Tests for data processing modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.chunker import Chunk, ChunkType, FinancialChunker
from data_processing.pdf_parser import PDFParser, ParsedDocument, TextSection


class TestChunk:
    """Tests for Chunk class."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            text="Apple reported revenue of $394.3 billion.",
            chunk_type=ChunkType.TEXT,
            metadata={"company": "AAPL"}
        )
        
        assert chunk.text == "Apple reported revenue of $394.3 billion."
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.metadata["company"] == "AAPL"
        assert chunk.chunk_id is not None
    
    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique."""
        chunk1 = Chunk(text="Text 1", chunk_type=ChunkType.TEXT, metadata={})
        chunk2 = Chunk(text="Text 2", chunk_type=ChunkType.TEXT, metadata={})
        
        assert chunk1.chunk_id != chunk2.chunk_id
    
    def test_chunk_token_count(self):
        """Test token count estimation."""
        chunk = Chunk(
            text="This is a test sentence.",
            chunk_type=ChunkType.TEXT,
            metadata={}
        )
        
        # Approximate: 24 chars / 4 = 6 tokens
        assert chunk.token_count == 6
    
    def test_chunk_to_dict(self):
        """Test serialization to dictionary."""
        chunk = Chunk(
            text="Test text",
            chunk_type=ChunkType.TEXT,
            metadata={"key": "value"}
        )
        
        data = chunk.to_dict()
        
        assert "chunk_id" in data
        assert data["text"] == "Test text"
        assert data["chunk_type"] == "text"
        assert data["metadata"]["key"] == "value"
    
    def test_chunk_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "chunk_id": "test123",
            "text": "Test text",
            "chunk_type": "table",
            "metadata": {"company": "MSFT"}
        }
        
        chunk = Chunk.from_dict(data)
        
        assert chunk.chunk_id == "test123"
        assert chunk.text == "Test text"
        assert chunk.chunk_type == ChunkType.TABLE
        assert chunk.metadata["company"] == "MSFT"


class TestFinancialChunker:
    """Tests for FinancialChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return FinancialChunker(chunk_size=100, overlap=20)
    
    def test_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
        assert chunker.preserve_tables is True
    
    def test_sentence_splitting(self, chunker):
        """Test sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
    
    def test_chunk_text(self, chunker):
        """Test basic text chunking."""
        text = "This is a test document. " * 50  # Create long text
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunk_type == ChunkType.TEXT for c in chunks)
    
    def test_chunk_metadata(self, chunker):
        """Test that metadata is preserved in chunks."""
        text = "Test text. " * 20
        metadata = {"company": "AAPL", "year": 2023}
        
        chunks = chunker.chunk_text(text, metadata=metadata)
        
        assert all(c.metadata["company"] == "AAPL" for c in chunks)
        assert all(c.metadata["year"] == 2023 for c in chunks)


class TestPDFParser:
    """Tests for PDFParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PDFParser()
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.extract_tables is True
        assert parser.identify_sections is True
    
    def test_section_patterns(self, parser):
        """Test that section patterns are defined."""
        assert "business" in parser.SECTION_PATTERNS
        assert "risk_factors" in parser.SECTION_PATTERNS
        assert "mda" in parser.SECTION_PATTERNS
    
    def test_clean_table(self, parser):
        """Test table cleaning."""
        table = [
            ["Header1", "Header2"],
            ["Value1", None],
            ["Value2", "Value3"]
        ]
        
        cleaned = parser._clean_table(table)
        
        assert cleaned[1][1] == ""  # None replaced with empty string
        assert all(len(row) == 2 for row in cleaned)


class TestNumericalExtraction:
    """Tests for numerical value extraction."""
    
    def test_extract_dollar_amounts(self):
        """Test extraction of dollar amounts."""
        from reasoning.numerical import NumericalReasoner
        
        reasoner = NumericalReasoner()
        text = "Revenue was $394.3 billion in 2022."
        
        numbers = reasoner.extract_numbers(text)
        
        assert len(numbers) > 0
        assert any(n.unit == "$" for n in numbers)
    
    def test_extract_percentages(self):
        """Test extraction of percentages."""
        from reasoning.numerical import NumericalReasoner
        
        reasoner = NumericalReasoner()
        text = "Growth rate was 15.5% year-over-year."
        
        numbers = reasoner.extract_numbers(text)
        
        assert len(numbers) > 0
        assert any(n.unit == "%" for n in numbers)
    
    def test_calculation_detection(self):
        """Test detection of calculation-requiring questions."""
        from reasoning.numerical import NumericalReasoner
        
        reasoner = NumericalReasoner()
        
        # Should require calculation
        needs_calc, _ = reasoner.requires_calculation("What was the revenue growth rate?")
        assert needs_calc is True
        
        # Should not require calculation
        needs_calc, _ = reasoner.requires_calculation("What is the company name?")
        assert needs_calc is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
