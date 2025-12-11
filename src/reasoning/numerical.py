"""
Numerical Reasoning Module
Financial Document Intelligence Platform

Handles numerical reasoning, calculations, and number extraction
from financial documents for accurate Q&A.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from loguru import logger


class OperationType(Enum):
    """Types of numerical operations."""
    GROWTH_RATE = "growth_rate"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    SUM = "sum"
    AVERAGE = "average"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"


@dataclass
class ExtractedNumber:
    """Represents an extracted number from text."""
    value: float
    original_text: str
    unit: Optional[str] = None
    scale: float = 1.0  # For billions, millions, etc.
    context: Optional[str] = None
    
    @property
    def scaled_value(self) -> float:
        return self.value * self.scale


@dataclass
class CalculationResult:
    """Result of a numerical calculation."""
    value: float
    operation: OperationType
    operands: List[ExtractedNumber]
    formula: str
    formatted_answer: str


class NumericalReasoner:
    """
    Handles numerical reasoning for financial Q&A.
    
    Features:
    - Detection of calculation-requiring questions
    - Number extraction from text and tables
    - Multiple operation types (growth rate, ratios, etc.)
    - Proper formatting of financial figures
    """
    
    # Keywords that indicate calculation is needed
    CALCULATION_KEYWORDS = {
        'growth': OperationType.GROWTH_RATE,
        'increase': OperationType.GROWTH_RATE,
        'decrease': OperationType.GROWTH_RATE,
        'change': OperationType.GROWTH_RATE,
        'percentage': OperationType.PERCENTAGE,
        'percent': OperationType.PERCENTAGE,
        '%': OperationType.PERCENTAGE,
        'ratio': OperationType.RATIO,
        'margin': OperationType.RATIO,
        'difference': OperationType.DIFFERENCE,
        'total': OperationType.SUM,
        'sum': OperationType.SUM,
        'average': OperationType.AVERAGE,
        'mean': OperationType.AVERAGE,
        'compared to': OperationType.DIFFERENCE,
        'year-over-year': OperationType.GROWTH_RATE,
        'yoy': OperationType.GROWTH_RATE,
        'quarter-over-quarter': OperationType.GROWTH_RATE,
        'qoq': OperationType.GROWTH_RATE,
    }
    
    # Scale keywords
    SCALE_PATTERNS = {
        r'\btrillion\b': 1e12,
        r'\bbillion\b': 1e9,
        r'\bmillion\b': 1e6,
        r'\bthousand\b': 1e3,
        r'\bT\b': 1e12,
        r'\bB\b': 1e9,
        r'\bM\b': 1e6,
        r'\bK\b': 1e3,
    }
    
    # Number extraction patterns
    NUMBER_PATTERNS = [
        # Currency amounts: $1,234.56 or $1.23 billion
        r'\$\s*([\d,]+\.?\d*)\s*(billion|million|thousand|B|M|K)?',
        # Percentages: 12.5% or 12.5 percent
        r'([\d,]+\.?\d*)\s*(%|percent)',
        # Plain numbers with optional scale: 1,234.56 billion
        r'([\d,]+\.?\d*)\s*(trillion|billion|million|thousand|T|B|M|K)?',
    ]
    
    def __init__(self, decimal_places: int = 2, use_commas: bool = True):
        """
        Initialize the numerical reasoner.
        
        Args:
            decimal_places: Number of decimal places for formatting
            use_commas: Whether to use thousand separators
        """
        self.decimal_places = decimal_places
        self.use_commas = use_commas
    
    def requires_calculation(self, question: str) -> Tuple[bool, Optional[OperationType]]:
        """
        Detect if question needs mathematical calculation.
        
        Args:
            question: The question text
            
        Returns:
            Tuple of (requires_calculation, operation_type)
        """
        question_lower = question.lower()
        
        for keyword, op_type in self.CALCULATION_KEYWORDS.items():
            if keyword in question_lower:
                return True, op_type
        
        # Check for comparison patterns
        if re.search(r'how much (more|less|higher|lower)', question_lower):
            return True, OperationType.DIFFERENCE
        
        if re.search(r'what is the .* rate', question_lower):
            return True, OperationType.GROWTH_RATE
        
        return False, None
    
    def extract_numbers(self,
                        text: str,
                        tables: List[pd.DataFrame] = None) -> List[ExtractedNumber]:
        """
        Extract relevant numbers from context.
        
        Args:
            text: Text content
            tables: Optional list of tables as DataFrames
            
        Returns:
            List of ExtractedNumber objects
        """
        numbers = []
        
        # Extract from text
        text_numbers = self._extract_from_text(text)
        numbers.extend(text_numbers)
        
        # Extract from tables
        if tables:
            for table in tables:
                table_numbers = self._extract_from_table(table)
                numbers.extend(table_numbers)
        
        return numbers
    
    def _extract_from_text(self, text: str) -> List[ExtractedNumber]:
        """Extract numbers from text content."""
        numbers = []
        
        # Find all number patterns
        for pattern in self.NUMBER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    # Extract the number value
                    num_str = match.group(1).replace(',', '')
                    value = float(num_str)
                    
                    # Check for scale
                    scale = 1.0
                    if len(match.groups()) > 1 and match.group(2):
                        scale_text = match.group(2).lower()
                        for scale_pattern, scale_value in self.SCALE_PATTERNS.items():
                            if re.search(scale_pattern, scale_text, re.IGNORECASE):
                                scale = scale_value
                                break
                    
                    # Determine unit
                    unit = None
                    if '$' in match.group(0):
                        unit = '$'
                    elif '%' in match.group(0) or 'percent' in match.group(0).lower():
                        unit = '%'
                    
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    numbers.append(ExtractedNumber(
                        value=value,
                        original_text=match.group(0),
                        unit=unit,
                        scale=scale,
                        context=context
                    ))
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to extract number from {match.group(0)}: {e}")
        
        return numbers
    
    def _extract_from_table(self, table: pd.DataFrame) -> List[ExtractedNumber]:
        """Extract numbers from a table."""
        numbers = []
        
        for col in table.columns:
            for idx, cell in table[col].items():
                cell_str = str(cell)
                
                # Try to extract number from cell
                cell_numbers = self._extract_from_text(cell_str)
                
                for num in cell_numbers:
                    # Add table context
                    num.context = f"Column: {col}, Row: {idx}"
                    numbers.append(num)
        
        return numbers
    
    def calculate(self,
                  operation: OperationType,
                  numbers: List[ExtractedNumber],
                  labels: List[str] = None) -> Optional[CalculationResult]:
        """
        Perform calculation based on operation type.
        
        Args:
            operation: Type of calculation to perform
            numbers: Numbers to use in calculation
            labels: Optional labels for the numbers
            
        Returns:
            CalculationResult or None if calculation failed
        """
        if len(numbers) < 1:
            return None
        
        try:
            if operation == OperationType.GROWTH_RATE:
                return self._calculate_growth_rate(numbers)
            elif operation == OperationType.PERCENTAGE:
                return self._calculate_percentage(numbers)
            elif operation == OperationType.RATIO:
                return self._calculate_ratio(numbers)
            elif operation == OperationType.DIFFERENCE:
                return self._calculate_difference(numbers)
            elif operation == OperationType.SUM:
                return self._calculate_sum(numbers)
            elif operation == OperationType.AVERAGE:
                return self._calculate_average(numbers)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return None
    
    def _calculate_growth_rate(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate growth rate between two values."""
        if len(numbers) < 2:
            return None
        
        # Assume first is new value, second is old value
        # Or try to determine from context (year mentions)
        new_value = numbers[0].scaled_value
        old_value = numbers[1].scaled_value
        
        if old_value == 0:
            return None
        
        growth_rate = ((new_value - old_value) / old_value) * 100
        
        formula = f"({new_value:,.2f} - {old_value:,.2f}) / {old_value:,.2f} × 100"
        formatted = self.format_answer(growth_rate, unit='%')
        
        return CalculationResult(
            value=growth_rate,
            operation=OperationType.GROWTH_RATE,
            operands=numbers[:2],
            formula=formula,
            formatted_answer=formatted
        )
    
    def _calculate_percentage(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate percentage of part to whole."""
        if len(numbers) < 2:
            return None
        
        part = numbers[0].scaled_value
        whole = numbers[1].scaled_value
        
        if whole == 0:
            return None
        
        percentage = (part / whole) * 100
        
        formula = f"({part:,.2f} / {whole:,.2f}) × 100"
        formatted = self.format_answer(percentage, unit='%')
        
        return CalculationResult(
            value=percentage,
            operation=OperationType.PERCENTAGE,
            operands=numbers[:2],
            formula=formula,
            formatted_answer=formatted
        )
    
    def _calculate_ratio(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate ratio between two values."""
        if len(numbers) < 2:
            return None
        
        numerator = numbers[0].scaled_value
        denominator = numbers[1].scaled_value
        
        if denominator == 0:
            return None
        
        ratio = numerator / denominator
        
        formula = f"{numerator:,.2f} / {denominator:,.2f}"
        formatted = f"{ratio:.2f}x"
        
        return CalculationResult(
            value=ratio,
            operation=OperationType.RATIO,
            operands=numbers[:2],
            formula=formula,
            formatted_answer=formatted
        )
    
    def _calculate_difference(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate difference between two values."""
        if len(numbers) < 2:
            return None
        
        value1 = numbers[0].scaled_value
        value2 = numbers[1].scaled_value
        
        difference = value1 - value2
        
        formula = f"{value1:,.2f} - {value2:,.2f}"
        unit = numbers[0].unit or numbers[1].unit
        formatted = self.format_answer(difference, unit=unit)
        
        return CalculationResult(
            value=difference,
            operation=OperationType.DIFFERENCE,
            operands=numbers[:2],
            formula=formula,
            formatted_answer=formatted
        )
    
    def _calculate_sum(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate sum of all values."""
        if len(numbers) < 1:
            return None
        
        values = [n.scaled_value for n in numbers]
        total = sum(values)
        
        formula = " + ".join(f"{v:,.2f}" for v in values)
        unit = numbers[0].unit if numbers else None
        formatted = self.format_answer(total, unit=unit)
        
        return CalculationResult(
            value=total,
            operation=OperationType.SUM,
            operands=numbers,
            formula=formula,
            formatted_answer=formatted
        )
    
    def _calculate_average(self, numbers: List[ExtractedNumber]) -> Optional[CalculationResult]:
        """Calculate average of all values."""
        if len(numbers) < 1:
            return None
        
        values = [n.scaled_value for n in numbers]
        average = sum(values) / len(values)
        
        formula = f"({' + '.join(f'{v:,.2f}' for v in values)}) / {len(values)}"
        unit = numbers[0].unit if numbers else None
        formatted = self.format_answer(average, unit=unit)
        
        return CalculationResult(
            value=average,
            operation=OperationType.AVERAGE,
            operands=numbers,
            formula=formula,
            formatted_answer=formatted
        )
    
    def format_answer(self,
                      value: float,
                      unit: str = None,
                      scale_label: str = None) -> str:
        """
        Format numerical answer with appropriate precision and units.
        
        Args:
            value: The numerical value
            unit: Unit symbol ($, %, etc.)
            scale_label: Scale label (billion, million, etc.)
            
        Returns:
            Formatted string
        """
        # Determine appropriate scale
        abs_value = abs(value)
        
        if scale_label is None:
            if abs_value >= 1e12:
                value = value / 1e12
                scale_label = "trillion"
            elif abs_value >= 1e9:
                value = value / 1e9
                scale_label = "billion"
            elif abs_value >= 1e6:
                value = value / 1e6
                scale_label = "million"
        
        # Format number
        if self.use_commas:
            formatted_num = f"{value:,.{self.decimal_places}f}"
        else:
            formatted_num = f"{value:.{self.decimal_places}f}"
        
        # Add unit and scale
        parts = []
        
        if unit == '$':
            parts.append(f"${formatted_num}")
        elif unit == '%':
            parts.append(f"{formatted_num}%")
        else:
            parts.append(formatted_num)
            if unit:
                parts.append(unit)
        
        if scale_label:
            parts.append(scale_label)
        
        return " ".join(parts)


def extract_and_calculate(question: str,
                          context: str,
                          tables: List[pd.DataFrame] = None) -> Optional[CalculationResult]:
    """
    Convenience function to extract numbers and perform calculation.
    
    Args:
        question: The question requiring calculation
        context: Text context containing numbers
        tables: Optional tables
        
    Returns:
        CalculationResult or None
    """
    reasoner = NumericalReasoner()
    
    # Check if calculation is needed
    needs_calc, op_type = reasoner.requires_calculation(question)
    
    if not needs_calc:
        return None
    
    # Extract numbers
    numbers = reasoner.extract_numbers(context, tables)
    
    if not numbers:
        return None
    
    # Perform calculation
    return reasoner.calculate(op_type, numbers)


if __name__ == "__main__":
    # Example usage
    reasoner = NumericalReasoner()
    
    # Test question detection
    questions = [
        "What was the revenue growth rate?",
        "What is Apple's revenue?",
        "How much did profit increase year-over-year?",
        "What is the debt-to-equity ratio?",
    ]
    
    print("Question Analysis:")
    for q in questions:
        needs_calc, op_type = reasoner.requires_calculation(q)
        print(f"  Q: {q}")
        print(f"     Needs calc: {needs_calc}, Type: {op_type}")
    
    # Test number extraction
    text = """
    Apple reported revenue of $394.3 billion in fiscal 2022, 
    compared to $365.8 billion in 2021. The gross margin was 43.3%.
    Operating expenses totaled $51.3 billion, an increase of 14%.
    """
    
    print("\nNumber Extraction:")
    numbers = reasoner.extract_numbers(text)
    for num in numbers:
        print(f"  {num.original_text}: {num.scaled_value:,.2f} (unit: {num.unit})")
    
    # Test calculation
    print("\nGrowth Rate Calculation:")
    result = reasoner.calculate(OperationType.GROWTH_RATE, numbers[:2])
    if result:
        print(f"  Formula: {result.formula}")
        print(f"  Result: {result.formatted_answer}")
