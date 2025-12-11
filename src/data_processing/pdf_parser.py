"""
PDF Parser Module
Financial Document Intelligence Platform

Extracts text and tables from SEC filings (10-K, 10-Q) while preserving structure.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pdfplumber
import pandas as pd
from loguru import logger


@dataclass
class TableData:
    """Represents an extracted table from a document."""
    data: pd.DataFrame
    page_number: int
    table_index: int
    caption: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert table to markdown format for embedding."""
        return self.data.to_markdown(index=False)
    
    def to_text(self) -> str:
        """Convert table to plain text format."""
        return self.data.to_string(index=False)


@dataclass
class TextSection:
    """Represents a text section from a document."""
    text: str
    page_number: int
    section_name: Optional[str] = None
    start_char: int = 0
    end_char: int = 0


@dataclass
class ParsedDocument:
    """Container for all parsed document data."""
    file_path: str
    text_sections: List[TextSection] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    full_text: str = ""
    
    @property
    def total_pages(self) -> int:
        return self.metadata.get('total_pages', 0)
    
    @property
    def company_name(self) -> Optional[str]:
        return self.metadata.get('company_name')


class PDFParser:
    """
    Parser for SEC filing PDFs.
    
    Extracts text and tables while preserving document structure,
    identifying key sections, and maintaining metadata.
    """
    
    # Common SEC filing section headers
    SECTION_PATTERNS = {
        'business': r'(?i)item\s*1[.\s]*business',
        'risk_factors': r'(?i)item\s*1a[.\s]*risk\s*factors',
        'properties': r'(?i)item\s*2[.\s]*properties',
        'legal_proceedings': r'(?i)item\s*3[.\s]*legal\s*proceedings',
        'mda': r"(?i)item\s*7[.\s]*management[']?s?\s*discussion",
        'financial_statements': r'(?i)item\s*8[.\s]*financial\s*statements',
        'controls': r'(?i)item\s*9a[.\s]*controls\s*and\s*procedures',
        'executive_compensation': r'(?i)item\s*11[.\s]*executive\s*compensation',
    }
    
    def __init__(self, 
                 extract_tables: bool = True,
                 identify_sections: bool = True,
                 min_table_rows: int = 2,
                 min_table_cols: int = 2):
        """
        Initialize the PDF parser.
        
        Args:
            extract_tables: Whether to extract tables separately
            identify_sections: Whether to identify SEC filing sections
            min_table_rows: Minimum rows for valid table
            min_table_cols: Minimum columns for valid table
        """
        self.extract_tables = extract_tables
        self.identify_sections = identify_sections
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        
    def parse_filing(self, pdf_path: str) -> ParsedDocument:
        """
        Parse SEC filing and return structured data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument containing text, tables, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {pdf_path}")
        
        parsed_doc = ParsedDocument(file_path=str(pdf_path))
        all_text_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                parsed_doc.metadata['total_pages'] = len(pdf.pages)
                parsed_doc.metadata['file_name'] = pdf_path.name
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    page_text = page.extract_text() or ""
                    
                    if page_text.strip():
                        text_section = TextSection(
                            text=page_text,
                            page_number=page_num
                        )
                        parsed_doc.text_sections.append(text_section)
                        all_text_parts.append(page_text)
                    
                    # Extract tables
                    if self.extract_tables:
                        tables = self._extract_page_tables(page, page_num)
                        parsed_doc.tables.extend(tables)
                
                # Combine all text
                parsed_doc.full_text = "\n\n".join(all_text_parts)
                
                # Identify sections
                if self.identify_sections:
                    self._identify_sections(parsed_doc)
                
                # Extract company info from first page
                self._extract_metadata(parsed_doc)
                
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
        
        logger.info(f"Parsed {len(parsed_doc.text_sections)} pages, "
                   f"{len(parsed_doc.tables)} tables")
        
        return parsed_doc
    
    def _extract_page_tables(self, page, page_num: int) -> List[TableData]:
        """Extract tables from a single page."""
        tables = []
        
        try:
            page_tables = page.extract_tables()
            
            for idx, table in enumerate(page_tables):
                if not table or len(table) < self.min_table_rows:
                    continue
                
                # Clean and validate table
                cleaned_table = self._clean_table(table)
                
                if cleaned_table is not None:
                    df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                    
                    if len(df.columns) >= self.min_table_cols:
                        table_data = TableData(
                            data=df,
                            page_number=page_num,
                            table_index=idx
                        )
                        tables.append(table_data)
                        
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _clean_table(self, table: List[List]) -> Optional[List[List]]:
        """Clean and validate extracted table data."""
        if not table:
            return None
        
        cleaned = []
        for row in table:
            if row:
                # Clean each cell
                cleaned_row = []
                for cell in row:
                    if cell is None:
                        cleaned_row.append("")
                    else:
                        # Clean whitespace and normalize
                        cleaned_cell = str(cell).strip()
                        cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                        cleaned_row.append(cleaned_cell)
                cleaned.append(cleaned_row)
        
        # Ensure consistent column count
        if cleaned:
            max_cols = max(len(row) for row in cleaned)
            cleaned = [row + [''] * (max_cols - len(row)) for row in cleaned]
        
        return cleaned if cleaned else None
    
    def _identify_sections(self, doc: ParsedDocument) -> None:
        """Identify SEC filing sections in the document."""
        sections_found = {}
        
        for section_name, pattern in self.SECTION_PATTERNS.items():
            match = re.search(pattern, doc.full_text)
            if match:
                sections_found[section_name] = {
                    'start': match.start(),
                    'header': match.group()
                }
        
        # Assign section names to text sections
        sorted_sections = sorted(sections_found.items(), 
                                key=lambda x: x[1]['start'])
        
        for text_section in doc.text_sections:
            for i, (section_name, section_info) in enumerate(sorted_sections):
                # Determine section boundaries
                start = section_info['start']
                end = (sorted_sections[i + 1][1]['start'] 
                      if i + 1 < len(sorted_sections) else len(doc.full_text))
                
                # Check if text section falls within this section
                # (simplified - could be more precise with character positions)
                if section_info['header'].lower() in text_section.text.lower():
                    text_section.section_name = section_name
                    break
        
        doc.metadata['sections'] = list(sections_found.keys())
    
    def _extract_metadata(self, doc: ParsedDocument) -> None:
        """Extract metadata like company name from document."""
        # Try to extract company name from first page
        first_page_text = doc.text_sections[0].text if doc.text_sections else ""
        
        # Common patterns for company names in SEC filings
        patterns = [
            r'(?i)(?:commission file number.*?\n)([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|LLC|Ltd|Company|Co)\.?)',
            r'(?i)([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|LLC|Ltd|Company|Co)\.?)\s*\n.*?form\s*10-[kq]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_page_text[:2000])
            if match:
                company_name = match.group(1).strip()
                doc.metadata['company_name'] = company_name
                break
        
        # Try to identify filing type
        if re.search(r'(?i)form\s*10-k', first_page_text[:1000]):
            doc.metadata['filing_type'] = '10-K'
        elif re.search(r'(?i)form\s*10-q', first_page_text[:1000]):
            doc.metadata['filing_type'] = '10-Q'
        
        # Try to extract fiscal year
        year_match = re.search(r'(?i)fiscal\s*year\s*(?:ended|ending)?\s*.*?(\d{4})', 
                               first_page_text[:2000])
        if year_match:
            doc.metadata['fiscal_year'] = int(year_match.group(1))
    
    def extract_tables_only(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract only tables from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of pandas DataFrames
        """
        doc = self.parse_filing(pdf_path)
        return [table.data for table in doc.tables]
    
    def get_sections(self, doc: ParsedDocument) -> Dict[str, str]:
        """
        Get identified sections from a parsed document.
        
        Args:
            doc: ParsedDocument instance
            
        Returns:
            Dict mapping section names to their text content
        """
        sections = {}
        
        for section_name in self.SECTION_PATTERNS.keys():
            section_texts = [
                ts.text for ts in doc.text_sections 
                if ts.section_name == section_name
            ]
            if section_texts:
                sections[section_name] = "\n\n".join(section_texts)
        
        return sections


# Utility functions
def parse_sec_filing(pdf_path: str, **kwargs) -> ParsedDocument:
    """Convenience function to parse an SEC filing."""
    parser = PDFParser(**kwargs)
    return parser.parse_filing(pdf_path)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file."""
    parser = PDFParser(extract_tables=False, identify_sections=False)
    doc = parser.parse_filing(pdf_path)
    return doc.full_text


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        parser = PDFParser()
        doc = parser.parse_filing(pdf_file)
        
        print(f"File: {doc.file_path}")
        print(f"Pages: {doc.total_pages}")
        print(f"Tables found: {len(doc.tables)}")
        print(f"Sections: {doc.metadata.get('sections', [])}")
        print(f"Company: {doc.company_name}")
    else:
        print("Usage: python pdf_parser.py <pdf_file>")
