"""Quantum Core module for SpiralMind-Nexus.

Implements quantum scoring algorithms including Fibonacci, Shannon entropy,
text complexity analysis, and S9 formula calculations.
"""

import math
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class QuantumResult:
    """Result of quantum analysis."""
    
    fibonacci_score: float
    entropy_score: float
    complexity_score: float
    s9_score: float
    quantum_score: float
    metadata: Dict[str, Any]


class QuantumCore:
    """Core quantum processing engine."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize QuantumCore.
        
        Args:
            config: Configuration dictionary for quantum processing
        """
        self.config = config or {}
        self.weights = self.config.get('weights', {
            'fibonacci': 0.3,
            'entropy': 0.25,
            'complexity': 0.25,
            's9': 0.2
        })
        self._fibonacci_cache = {0: 0, 1: 1}
        
    def fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number with memoization.
        
        Args:
            n: Position in Fibonacci sequence
            
        Returns:
            Fibonacci number at position n
        """
        if n < 0:
            return 0
            
        if n in self._fibonacci_cache:
            return self._fibonacci_cache[n]
            
        # Calculate iteratively to avoid stack overflow
        a, b = 0, 1
        for i in range(2, n + 1):
            if i not in self._fibonacci_cache:
                a, b = b, a + b
                self._fibonacci_cache[i] = b
            else:
                a, b = self._fibonacci_cache[i-1], self._fibonacci_cache[i]
                
        return self._fibonacci_cache[n]
    
    def calculate_fibonacci_score(self, text: str) -> float:
        """Calculate Fibonacci-based score for text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Normalized Fibonacci score (0.0 to 1.0)
        """
        if not text:
            return 0.0
            
        length = len(text.replace(' ', ''))
        if length == 0:
            return 0.0
            
        # Find closest Fibonacci number
        fib_pos = 1
        while self.fibonacci(fib_pos) < length:
            fib_pos += 1
            
        fib_val = self.fibonacci(fib_pos)
        
        # Calculate alignment score
        if fib_val == 0:
            return 0.0
            
        alignment = 1.0 - abs(length - fib_val) / fib_val
        return max(0.0, min(1.0, alignment))
    
    def calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Shannon entropy value
        """
        if not text:
            return 0.0
            
        # Count character frequencies
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
            
        length = len(text)
        if length <= 1:
            return 0.0
            
        # Calculate entropy
        entropy = 0.0
        for count in freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
                
        return entropy
    
    def calculate_entropy_score(self, text: str) -> float:
        """Calculate normalized entropy score.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Normalized entropy score (0.0 to 1.0)
        """
        entropy = self.calculate_shannon_entropy(text)
        
        # Normalize entropy (max possible entropy for text length)
        unique_chars = len(set(text)) if text else 1
        max_entropy = math.log2(unique_chars) if unique_chars > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity based on various metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not text:
            return 0.0
            
        # Word count and average word length
        words = text.split()
        if not words:
            return 0.0
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence complexity
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else len(words)
        
        # Character diversity
        char_diversity = len(set(text.lower())) / len(text) if text else 0
        
        # Punctuation density
        punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}"\'-')
        punctuation_density = punctuation_count / len(text) if text else 0
        
        # Combine metrics
        complexity = (
            min(avg_word_length / 10, 1.0) * 0.3 +
            min(avg_sentence_length / 20, 1.0) * 0.3 +
            char_diversity * 0.2 +
            min(punctuation_density * 10, 1.0) * 0.2
        )
        
        return min(1.0, complexity)
    
    def calculate_s9_formula(self, text: str) -> float:
        """Calculate S9 formula score.
        
        The S9 formula combines multiple text metrics into a single score.
        
        Args:
            text: Input text to analyze
            
        Returns:
            S9 formula score (0.0 to 1.0)
        """
        if not text:
            return 0.0
            
        # Text metrics
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
            
        # Calculate components
        avg_word_length = sum(len(word) for word in words) / word_count
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / word_count
        
        # Readability approximation
        sentences = len(re.split(r'[.!?]+', text))
        if sentences == 0:
            sentences = 1
            
        avg_sentence_length = word_count / sentences
        
        # S9 formula calculation
        s9 = (
            (avg_word_length / 10) * 0.2 +
            lexical_diversity * 0.3 +
            min(avg_sentence_length / 15, 1.0) * 0.2 +
            min(length / 1000, 1.0) * 0.3
        )
        
        return min(1.0, s9)
    
    def calculate_quantum_score(self, 
                              fibonacci_score: float,
                              entropy_score: float,
                              complexity_score: float,
                              s9_score: float) -> float:
        """Calculate combined quantum score.
        
        Args:
            fibonacci_score: Fibonacci alignment score
            entropy_score: Shannon entropy score
            complexity_score: Text complexity score
            s9_score: S9 formula score
            
        Returns:
            Combined quantum score (0.0 to 1.0)
        """
        quantum_score = (
            fibonacci_score * self.weights['fibonacci'] +
            entropy_score * self.weights['entropy'] +
            complexity_score * self.weights['complexity'] +
            s9_score * self.weights['s9']
        )
        
        return min(1.0, quantum_score)
    
    def process(self, text: str, metadata: Dict[str, Any] = None) -> QuantumResult:
        """Process text through quantum analysis.
        
        Args:
            text: Input text to analyze
            metadata: Additional metadata for analysis
            
        Returns:
            QuantumResult with all calculated scores
        """
        logger.debug(f"Processing text of length {len(text)}")
        
        if not text:
            return QuantumResult(
                fibonacci_score=0.0,
                entropy_score=0.0,
                complexity_score=0.0,
                s9_score=0.0,
                quantum_score=0.0,
                metadata=metadata or {}
            )
        
        # Calculate individual scores
        fibonacci_score = self.calculate_fibonacci_score(text)
        entropy_score = self.calculate_entropy_score(text)
        complexity_score = self.calculate_text_complexity(text)
        s9_score = self.calculate_s9_formula(text)
        
        # Calculate combined quantum score
        quantum_score = self.calculate_quantum_score(
            fibonacci_score, entropy_score, complexity_score, s9_score
        )
        
        result = QuantumResult(
            fibonacci_score=fibonacci_score,
            entropy_score=entropy_score,
            complexity_score=complexity_score,
            s9_score=s9_score,
            quantum_score=quantum_score,
            metadata={
                'text_length': len(text),
                'word_count': len(text.split()),
                'weights_used': self.weights.copy(),
                **(metadata or {})
            }
        )
        
        logger.debug(f"Quantum processing complete: score={quantum_score:.3f}")
        return result
    
    def batch_process(self, texts: List[str]) -> List[QuantumResult]:
        """Process multiple texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of QuantumResult objects
        """
        logger.info(f"Batch processing {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.process(text, {'batch_index': i})
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                # Return zero result for failed processing
                results.append(QuantumResult(
                    fibonacci_score=0.0,
                    entropy_score=0.0,
                    complexity_score=0.0,
                    s9_score=0.0,
                    quantum_score=0.0,
                    metadata={'batch_index': i, 'error': str(e)}
                ))
                
        return results
