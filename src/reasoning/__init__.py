"""
Reasoning Module
Financial Document Intelligence Platform

This module handles:
- Numerical reasoning and calculations
- Citation extraction and formatting
"""

from .numerical import NumericalReasoner
from .citations import CitationManager, Citation

__all__ = [
    'NumericalReasoner',
    'CitationManager',
    'Citation'
]
