"""
Data Processing Module
Financial Document Intelligence Platform

This module handles:
- PDF parsing and text extraction
- Document chunking with semantic preservation
- SEC filing downloads from EDGAR
"""

from .pdf_parser import PDFParser
from .chunker import FinancialChunker, Chunk
from .sec_downloader import SECDownloader

__all__ = [
    'PDFParser',
    'FinancialChunker', 
    'Chunk',
    'SECDownloader'
]
