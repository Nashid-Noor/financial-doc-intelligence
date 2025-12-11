"""
Evaluation Module
Financial Document Intelligence Platform

Comprehensive evaluation framework for the RAG system.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    question: str
    predicted_answer: str
    reference_answer: str
    exact_match: bool
    f1_score: float
    numerical_accuracy: Optional[bool]
    retrieval_precision: float
    citation_accuracy: float


class EvaluationMetrics:
    """
    Metrics calculator for financial Q&A evaluation.
    """
    
    def __init__(self, numerical_tolerance: float = 0.05):
        """
        Initialize metrics calculator.
        
        Args:
            numerical_tolerance: Tolerance for numerical accuracy (default 5%)
        """
        self.numerical_tolerance = numerical_tolerance
    
    def calculate_exact_match(self, 
                              predictions: List[str], 
                              references: List[str]) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Exact match accuracy (0-1)
        """
        if not predictions or not references:
            return 0.0
        
        matches = sum(
            self._normalize_answer(p) == self._normalize_answer(r)
            for p, r in zip(predictions, references)
        )
        
        return matches / len(predictions)
    
    def calculate_f1(self, 
                     predictions: List[str], 
                     references: List[str]) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Average F1 score (0-1)
        """
        if not predictions or not references:
            return 0.0
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            f1 = self._token_f1(pred, ref)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _token_f1(self, prediction: str, reference: str) -> float:
        """Calculate F1 between two strings."""
        pred_tokens = set(self._normalize_answer(prediction).split())
        ref_tokens = set(self._normalize_answer(reference).split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_numerical_accuracy(self,
                                     predictions: List[str],
                                     references: List[str],
                                     tolerance: float = None) -> float:
        """
        Calculate accuracy for numerical answers.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            tolerance: Tolerance percentage (default: 5%)
            
        Returns:
            Numerical accuracy (0-1)
        """
        tolerance = tolerance or self.numerical_tolerance
        
        correct = 0
        numerical_count = 0
        
        for pred, ref in zip(predictions, references):
            ref_nums = self._extract_numbers(ref)
            
            if not ref_nums:
                continue
            
            numerical_count += 1
            pred_nums = self._extract_numbers(pred)
            
            if not pred_nums:
                continue
            
            # Check if any predicted number matches reference
            for ref_num in ref_nums:
                for pred_num in pred_nums:
                    if ref_num != 0 and abs(pred_num - ref_num) / abs(ref_num) <= tolerance:
                        correct += 1
                        break
                else:
                    continue
                break
        
        return correct / numerical_count if numerical_count > 0 else 0.0
    
    def calculate_retrieval_metrics(self,
                                    retrieved: List[List[str]],
                                    relevant: List[List[str]],
                                    k: int = 5) -> Dict[str, float]:
        """
        Calculate retrieval metrics (Precision@K, Recall@K, MRR).
        
        Args:
            retrieved: List of retrieved document IDs per query
            relevant: List of relevant document IDs per query
            k: Cutoff for Precision@K and Recall@K
            
        Returns:
            Dict with precision, recall, and MRR
        """
        precisions = []
        recalls = []
        mrrs = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_set = set(ret[:k])
            rel_set = set(rel)
            
            # Precision@K
            if ret_set:
                precision = len(ret_set & rel_set) / len(ret_set)
            else:
                precision = 0.0
            precisions.append(precision)
            
            # Recall@K
            if rel_set:
                recall = len(ret_set & rel_set) / len(rel_set)
            else:
                recall = 0.0
            recalls.append(recall)
            
            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(ret[:k]):
                if doc_id in rel_set:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)
        
        return {
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls),
            'mrr': np.mean(mrrs)
        }
    
    def calculate_citation_accuracy(self,
                                    pred_citations: List[List[str]],
                                    gold_citations: List[List[str]]) -> float:
        """
        Calculate citation attribution accuracy.
        
        Args:
            pred_citations: Predicted citations per answer
            gold_citations: Gold standard citations per answer
            
        Returns:
            Citation accuracy (0-1)
        """
        if not pred_citations or not gold_citations:
            return 0.0
        
        accuracies = []
        for pred, gold in zip(pred_citations, gold_citations):
            pred_set = set(pred)
            gold_set = set(gold)
            
            if gold_set:
                accuracy = len(pred_set & gold_set) / len(gold_set)
            else:
                accuracy = 1.0 if not pred_set else 0.0
            
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase
        answer = answer.lower()
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        return answer
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Match numbers including decimals and percentages
        pattern = r'[\d,]+\.?\d*'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                num = float(match.replace(',', ''))
                numbers.append(num)
            except ValueError:
                continue
        
        return numbers


class RAGEvaluator:
    """
    End-to-end evaluator for the RAG system.
    """
    
    def __init__(self, retriever, qa_model=None):
        """
        Initialize evaluator.
        
        Args:
            retriever: HybridRetriever instance
            qa_model: FinancialQAModel instance (optional)
        """
        self.retriever = retriever
        self.qa_model = qa_model
        self.metrics = EvaluationMetrics()
    
    def evaluate_dataset(self, 
                         test_data: List[Dict],
                         top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate on a test dataset.
        
        Args:
            test_data: List of test examples with 'question', 'context', 'answer'
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with all evaluation metrics
        """
        predictions = []
        references = []
        
        for example in test_data:
            question = example['question']
            reference = example['answer']
            
            # Retrieve relevant chunks
            results = self.retriever.retrieve(question, top_k=top_k)
            chunks = [r.chunk for r in results]
            
            # Generate answer (mock if no model)
            if self.qa_model:
                prediction = self.qa_model.generate_answer(question, chunks)
            else:
                # Mock prediction using retrieved context
                prediction = chunks[0].text if chunks else ""
            
            predictions.append(prediction)
            references.append(reference)
        
        # Calculate metrics
        results = {
            'exact_match': self.metrics.calculate_exact_match(predictions, references),
            'f1_score': self.metrics.calculate_f1(predictions, references),
            'numerical_accuracy': self.metrics.calculate_numerical_accuracy(predictions, references),
        }
        
        return results
    
    def run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict:
        """
        Run comprehensive evaluation with detailed breakdown.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Comprehensive evaluation results
        """
        # Group by question type
        extraction_data = [d for d in test_data if d.get('type') == 'extraction']
        calculation_data = [d for d in test_data if d.get('type') == 'calculation']
        
        results = {
            'overall': self.evaluate_dataset(test_data),
            'by_type': {
                'extraction': self.evaluate_dataset(extraction_data) if extraction_data else {},
                'calculation': self.evaluate_dataset(calculation_data) if calculation_data else {}
            },
            'num_examples': len(test_data)
        }
        
        return results


def load_finqa_test_set(path: str) -> List[Dict]:
    """Load FinQA test set."""
    with open(path) as f:
        data = json.load(f)
    
    formatted = []
    for item in data:
        formatted.append({
            'question': item['qa']['question'],
            'context': item.get('text', ''),
            'answer': str(item['qa']['exe_ans']),
            'type': 'calculation' if item['qa'].get('program') else 'extraction'
        })
    
    return formatted


def run_baseline_comparison(evaluator: RAGEvaluator, test_data: List[Dict]) -> Dict:
    """
    Compare against baselines.
    
    Returns comparison table data.
    """
    results = {}
    
    # Full system evaluation
    results['full_system'] = evaluator.evaluate_dataset(test_data)
    
    # Note: Add baseline comparisons here when models are available
    # - GPT-4 zero-shot
    # - GPT-4 + RAG
    # - Fine-tuned Llama without RAG
    
    return results


if __name__ == "__main__":
    # Example usage
    metrics = EvaluationMetrics()
    
    # Test exact match
    predictions = ["$394.3 billion", "15.5%", "Apple Inc."]
    references = ["$394.3 billion", "15.5%", "Apple"]
    
    em = metrics.calculate_exact_match(predictions, references)
    f1 = metrics.calculate_f1(predictions, references)
    num_acc = metrics.calculate_numerical_accuracy(predictions, references)
    
    print("Evaluation Metrics Example:")
    print(f"  Exact Match: {em:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Numerical Accuracy: {num_acc:.2%}")
    
    # Test retrieval metrics
    retrieved = [["doc1", "doc2", "doc3"], ["doc2", "doc4", "doc5"]]
    relevant = [["doc1", "doc3"], ["doc2", "doc3"]]
    
    ret_metrics = metrics.calculate_retrieval_metrics(retrieved, relevant, k=3)
    print("\nRetrieval Metrics:")
    for key, value in ret_metrics.items():
        print(f"  {key}: {value:.2%}")
