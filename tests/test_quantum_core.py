"""Tests for Quantum Core functionality."""

from spiral.core.quantum_core import QuantumCore, QuantumResult


class TestQuantumCore:
    """Test cases for QuantumCore."""
    
    def test_fibonacci_calculation(self, quantum_core):
        """Test Fibonacci number calculation."""
        # Test basic Fibonacci numbers
        assert quantum_core.fibonacci(0) == 0
        assert quantum_core.fibonacci(1) == 1
        assert quantum_core.fibonacci(2) == 1
        assert quantum_core.fibonacci(3) == 2
        assert quantum_core.fibonacci(4) == 3
        assert quantum_core.fibonacci(5) == 5
        assert quantum_core.fibonacci(6) == 8
        assert quantum_core.fibonacci(10) == 55
        
        # Test edge cases
        assert quantum_core.fibonacci(-1) == 0
        assert quantum_core.fibonacci(-10) == 0
    
    def test_fibonacci_score_calculation(self, quantum_core):
        """Test Fibonacci score calculation."""
        # Test empty text
        assert quantum_core.calculate_fibonacci_score("") == 0.0
        
        # Test single character (length 1, closest Fibonacci is 1)
        score = quantum_core.calculate_fibonacci_score("A")
        assert 0.0 <= score <= 1.0
        
        # Test text with length 5 (Fibonacci number)
        score = quantum_core.calculate_fibonacci_score("Hello")
        assert 0.0 <= score <= 1.0
        
        # Test longer text
        score = quantum_core.calculate_fibonacci_score("This is a longer text for testing")
        assert 0.0 <= score <= 1.0
    
    def test_shannon_entropy_calculation(self, quantum_core):
        """Test Shannon entropy calculation."""
        # Test empty text
        assert quantum_core.calculate_shannon_entropy("") == 0.0
        
        # Test single character
        assert quantum_core.calculate_shannon_entropy("A") == 0.0
        
        # Test repeated characters (low entropy)
        entropy = quantum_core.calculate_shannon_entropy("AAAA")
        assert entropy == 0.0
        
        # Test mixed characters (higher entropy)
        entropy = quantum_core.calculate_shannon_entropy("ABCD")
        assert entropy == 2.0  # log2(4) for 4 equally likely characters
        
        # Test realistic text
        entropy = quantum_core.calculate_shannon_entropy("Hello, World!")
        assert entropy > 0.0
    
    def test_entropy_score_calculation(self, quantum_core):
        """Test entropy score normalization."""
        # Test empty text
        assert quantum_core.calculate_entropy_score("") == 0.0
        
        # Test various texts
        texts = [
            "A",
            "Hello",
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog"
        ]
        
        for text in texts:
            score = quantum_core.calculate_entropy_score(text)
            assert 0.0 <= score <= 1.0
    
    def test_text_complexity_calculation(self, quantum_core):
        """Test text complexity calculation."""
        # Test empty text
        assert quantum_core.calculate_text_complexity("") == 0.0
        
        # Test simple text
        complexity = quantum_core.calculate_text_complexity("Hello")
        assert 0.0 <= complexity <= 1.0
        
        # Test complex text with punctuation
        complex_text = "This is a complex sentence with punctuation, numbers (123), and various symbols!"
        complexity = quantum_core.calculate_text_complexity(complex_text)
        assert 0.0 <= complexity <= 1.0
        
        # Complex text should have higher complexity than simple text
        simple_complexity = quantum_core.calculate_text_complexity("hello world")
        assert complexity > simple_complexity
    
    def test_s9_formula_calculation(self, quantum_core):
        """Test S9 formula calculation."""
        # Test empty text
        assert quantum_core.calculate_s9_formula("") == 0.0
        
        # Test various texts
        texts = [
            "Hello",
            "This is a test.",
            "The quick brown fox jumps over the lazy dog. This is a longer sentence with more complexity.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt."
        ]
        
        for text in texts:
            score = quantum_core.calculate_s9_formula(text)
            assert 0.0 <= score <= 1.0
    
    def test_quantum_score_calculation(self, quantum_core):
        """Test combined quantum score calculation."""
        # Test with known values
        fibonacci_score = 0.8
        entropy_score = 0.6
        complexity_score = 0.7
        s9_score = 0.5
        
        quantum_score = quantum_core.calculate_quantum_score(
            fibonacci_score, entropy_score, complexity_score, s9_score
        )
        
        # Should be weighted average
        expected = (
            fibonacci_score * 0.3 +
            entropy_score * 0.25 +
            complexity_score * 0.25 +
            s9_score * 0.2
        )
        
        assert abs(quantum_score - expected) < 0.001
        assert 0.0 <= quantum_score <= 1.0
    
    def test_process_method(self, quantum_core, sample_texts):
        """Test main process method."""
        for text in sample_texts:
            result = quantum_core.process(text)
            
            # Check result structure
            assert isinstance(result, QuantumResult)
            assert hasattr(result, 'fibonacci_score')
            assert hasattr(result, 'entropy_score')
            assert hasattr(result, 'complexity_score')
            assert hasattr(result, 's9_score')
            assert hasattr(result, 'quantum_score')
            assert hasattr(result, 'metadata')
            
            # Check score ranges
            assert 0.0 <= result.fibonacci_score <= 1.0
            assert 0.0 <= result.entropy_score <= 1.0
            assert 0.0 <= result.complexity_score <= 1.0
            assert 0.0 <= result.s9_score <= 1.0
            assert 0.0 <= result.quantum_score <= 1.0
            
            # Check metadata
            assert 'text_length' in result.metadata
            assert 'word_count' in result.metadata
            assert 'weights_used' in result.metadata
    
    def test_batch_process(self, quantum_core, sample_texts):
        """Test batch processing."""
        results = quantum_core.batch_process(sample_texts)
        
        assert len(results) == len(sample_texts)
        
        for i, result in enumerate(results):
            assert isinstance(result, QuantumResult)
            assert result.metadata['batch_index'] == i
            assert 0.0 <= result.quantum_score <= 1.0
    
    def test_empty_batch_process(self, quantum_core):
        """Test batch processing with empty list."""
        results = quantum_core.batch_process([])
        assert results == []
    
    def test_process_with_metadata(self, quantum_core):
        """Test processing with custom metadata."""
        text = "Test text"
        custom_metadata = {'test_key': 'test_value', 'number': 42}
        
        result = quantum_core.process(text, metadata=custom_metadata)
        
        # Custom metadata should be included
        assert 'test_key' in result.metadata
        assert 'number' in result.metadata
        assert result.metadata['test_key'] == 'test_value'
        assert result.metadata['number'] == 42
        
        # Default metadata should still be present
        assert 'text_length' in result.metadata
        assert 'word_count' in result.metadata
    
    def test_weights_configuration(self):
        """Test quantum core with custom weights."""
        custom_weights = {
            'fibonacci': 0.4,
            'entropy': 0.3,
            'complexity': 0.2,
            's9': 0.1
        }
        
        custom_config = {'weights': custom_weights}
        quantum_core = QuantumCore(custom_config)
        
        assert quantum_core.weights == custom_weights
        
        # Test that custom weights are used in calculation
        result = quantum_core.process("Test text")
        assert result.metadata['weights_used'] == custom_weights
    
    def test_fibonacci_caching(self, quantum_core):
        """Test Fibonacci number caching."""
        # Calculate same Fibonacci number multiple times
        n = 20
        
        result1 = quantum_core.fibonacci(n)
        result2 = quantum_core.fibonacci(n)
        result3 = quantum_core.fibonacci(n)
        
        # Should return same result
        assert result1 == result2 == result3
        
        # Should have cached the value
        assert n in quantum_core._fibonacci_cache
        assert quantum_core._fibonacci_cache[n] == result1
    
    def test_error_handling(self, quantum_core):
        """Test error handling in batch processing."""
        # This test would need to simulate an error condition
        # For now, we test with valid inputs
        texts = ["valid text", "another valid text"]
        results = quantum_core.batch_process(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, QuantumResult)
            assert 'error' not in result.metadata
