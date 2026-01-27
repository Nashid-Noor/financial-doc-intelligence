"""
Model Module
Financial Document Intelligence Platform

This module handles:
- Fine-tuning Llama models with QLoRA
- Model inference and generation
"""

# from .fine_tune import FinancialQATrainer, TrainingConfig
from .inference import FinancialQAModel

__all__ = [
    # 'FinancialQATrainer',
    # 'TrainingConfig', 
    'FinancialQAModel'
]
