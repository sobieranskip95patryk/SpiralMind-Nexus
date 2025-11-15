"""Tests for Synergy Orchestrator and integration."""

import pytest
from spiral.pipeline.synergy_orchestrator import SynergyOrchestrator, ProcessingRequest, ProcessingResult
from spiral.pipeline.double_pipeline import execute, batch_execute, create_event


class TestSynergyOrchestrator:
    """Test cases for SynergyOrchestrator."""
    
    def test_initialization(self, test_config):
        """Test orchestrator initialization."""
        orchestrator = SynergyOrchestrator(test_config)
        
        assert orchestrator.config == test_config
        assert orchestrator.quantum_core is not None
        assert orchestrator.gokai_calculator is not None
        assert len(orchestrator.stats) > 0
        assert orchestrator._result_cache == {}
    
    def test_process_single_text(self, synergy_orchestrator):
        """Test processing single text."""
        text = "This is a test text for processing."
        context = {'test': True}
        
        result = synergy_orchestrator.process(text, context)
        
        assert isinstance(result, ProcessingResult)
        assert result.quantum_result is not None
        assert result.gokai_score is not None
        assert result.decision in ['accept', 'reject', 'review', 'unknown']
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0
        assert 'pipeline_mode' in result.metadata
    
    def test_process_empty_text(self, synergy_orchestrator):
        """Test processing empty text."""
        result = synergy_orchestrator.process("")
        
        assert isinstance(result, ProcessingResult)
        assert result.quantum_result.quantum_score == 0.0
        assert result.gokai_score.value == 0.0
        assert result.decision in ['reject', 'unknown']
    
    def test_decision_making_accept(self, synergy_orchestrator):
        """Test decision making for high-quality text."""
        # Create a text that should get high scores
        text = "This is a high-quality, well-structured text with excellent complexity and entropy characteristics that should receive positive scoring."
        context = {'importance': 0.9, 'quality': 0.95}
        
        result = synergy_orchestrator.process(text, context)
        
        # Should likely be accepted or at least reviewed
        assert result.decision in ['accept', 'review']
        assert result.confidence > 0.0
    
    def test_decision_making_reject(self, synergy_orchestrator):
        """Test decision making for low-quality text."""
        # Create a very simple text that should get low scores
        text = "a"
        context = {'importance': 0.1}
        
        result = synergy_orchestrator.process(text, context)
        
        # Should likely be rejected or reviewed
        assert result.decision in ['reject', 'review']
    
    def test_batch_processing_sequential(self, synergy_orchestrator, sample_texts):
        """Test sequential batch processing."""
        requests = [ProcessingRequest(text=text) for text in sample_texts[:3]]
        
        results = synergy_orchestrator.batch_process(requests, parallel=False)
        
        assert len(results) == len(requests)
        
        for i, result in enumerate(results):
            assert isinstance(result, ProcessingResult)
            assert result.metadata['request_id'] == requests[i].request_id
    
    def test_batch_processing_parallel(self, synergy_orchestrator, sample_texts):
        """Test parallel batch processing."""
        requests = [ProcessingRequest(text=text) for text in sample_texts[:3]]
        
        results = synergy_orchestrator.batch_process(requests, parallel=True)
        
        assert len(results) == len(requests)
        
        for i, result in enumerate(results):
            assert isinstance(result, ProcessingResult)
            assert result.metadata['request_id'] == requests[i].request_id
    
    @pytest.mark.asyncio
    async def test_async_processing(self, synergy_orchestrator):
        """Test asynchronous processing."""
        text = "Async test text"
        
        result = await synergy_orchestrator.process_async(text)
        
        assert isinstance(result, ProcessingResult)
        assert result.quantum_result is not None
        assert result.gokai_score is not None
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, synergy_orchestrator, sample_texts):
        """Test asynchronous batch processing."""
        requests = [ProcessingRequest(text=text) for text in sample_texts[:3]]
        
        results = await synergy_orchestrator.batch_process_async(requests)
        
        assert len(results) == len(requests)
        
        for result in results:
            assert isinstance(result, ProcessingResult)
    
    def test_caching(self, synergy_orchestrator):
        """Test result caching."""
        text = "Test text for caching"
        context = {'test': 'cache'}
        
        # First processing
        result1 = synergy_orchestrator.process(text, context, use_cache=True)
        
        # Second processing should use cache
        result2 = synergy_orchestrator.process(text, context, use_cache=True)
        
        # Results should be identical (from cache)
        assert result1.quantum_result.quantum_score == result2.quantum_result.quantum_score
        assert result1.gokai_score.value == result2.gokai_score.value
        assert result1.decision == result2.decision
    
    def test_statistics_tracking(self, synergy_orchestrator, sample_texts):
        """Test statistics tracking."""
        # Get initial stats
        initial_stats = synergy_orchestrator.get_statistics()
        initial_count = initial_stats['total_processed']
        
        # Process some texts
        for text in sample_texts[:3]:
            synergy_orchestrator.process(text)
        
        # Get updated stats
        updated_stats = synergy_orchestrator.get_statistics()
        
        assert updated_stats['total_processed'] == initial_count + 3
        assert updated_stats['successful_processed'] >= initial_stats.get('successful_processed', 0)
        assert 'success_rate' in updated_stats
        assert 'average_processing_time' in updated_stats
    
    def test_cache_management(self, synergy_orchestrator):
        """Test cache clearing and management."""
        # Process some texts to populate cache
        texts = ["Text 1", "Text 2", "Text 3"]
        for text in texts:
            synergy_orchestrator.process(text, use_cache=True)
        
        # Check cache has entries
        assert len(synergy_orchestrator._result_cache) > 0
        
        # Clear cache
        synergy_orchestrator.clear_cache()
        
        # Check cache is empty
        assert len(synergy_orchestrator._result_cache) == 0
    
    def test_reset_statistics(self, synergy_orchestrator):
        """Test statistics reset."""
        # Process some texts
        synergy_orchestrator.process("Test text")
        
        # Check stats are not zero
        stats = synergy_orchestrator.get_statistics()
        assert stats['total_processed'] > 0
        
        # Reset statistics
        synergy_orchestrator.reset_statistics()
        
        # Check stats are reset
        reset_stats = synergy_orchestrator.get_statistics()
        assert reset_stats['total_processed'] == 0
        assert reset_stats['successful_processed'] == 0
        assert reset_stats['failed_processed'] == 0
    
    def test_context_influence(self, synergy_orchestrator):
        """Test how context influences decisions."""
        text = "Medium quality text for testing context influence"
        
        # Process with high importance
        high_importance_result = synergy_orchestrator.process(
            text, {'importance': 0.9, 'urgency': 0.8}
        )
        
        # Process with low importance
        low_importance_result = synergy_orchestrator.process(
            text, {'importance': 0.1, 'urgency': 0.1}
        )
        
        # Context should influence the results
        assert high_importance_result.confidence != low_importance_result.confidence or \
               high_importance_result.decision != low_importance_result.decision


class TestDoublePipeline:
    """Test cases for double pipeline functions."""
    
    def test_execute_basic(self):
        """Test basic execute function."""
        text = "This is a test text for the pipeline."
        
        result = execute(text)
        
        assert result['success'] is True
        assert 'result' in result
        assert 'decision' in result
        assert 'confidence' in result
        assert 'quantum_score' in result
        assert 'gokai_score' in result
        assert 'metadata' in result
    
    def test_execute_with_context(self):
        """Test execute with context."""
        text = "Test text with context"
        context = {'importance': 0.8, 'source': 'test'}
        
        result = execute(text, context=context, mode="gokai")
        
        assert result['success'] is True
        assert result['metadata']['mode'] == 'gokai'
    
    def test_execute_invalid_text(self):
        """Test execute with invalid text."""
        result = execute("")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_batch_execute(self, sample_texts):
        """Test batch execute function."""
        texts = sample_texts[:3]  # Use first 3 texts
        
        results = batch_execute(texts, parallel=False)
        
        assert len(results) == len(texts)
        
        for result in results:
            # Handle empty text case
            if any(text == "" for text in texts):
                # Empty text should fail
                continue
            assert 'success' in result
            assert 'metadata' in result
    
    def test_batch_execute_parallel(self, sample_texts):
        """Test parallel batch execute."""
        texts = [text for text in sample_texts[:3] if text.strip()]  # Remove empty texts
        
        results = batch_execute(texts, parallel=True)
        
        assert len(results) == len(texts)
        
        for result in results:
            assert 'success' in result
    
    def test_create_event(self):
        """Test event creation."""
        text = "Test event text"
        
        result = create_event(
            text=text,
            event_type="test_event",
            metadata={'test': True},
            process_immediately=True
        )
        
        assert result['success'] is True
        assert 'event' in result
        assert result['event']['type'] == "test_event"
        assert result['event']['processed'] is True
    
    def test_create_event_no_processing(self):
        """Test event creation without immediate processing."""
        text = "Test event text"
        
        result = create_event(
            text=text,
            event_type="delayed_event",
            process_immediately=False
        )
        
        assert result['success'] is True
        assert 'event' in result
        assert result['event']['processed'] is False
    
    def test_create_event_invalid_text(self):
        """Test event creation with invalid text."""
        result = create_event(text="")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_memory_integration(self):
        """Test memory integration in pipeline."""
        text = "Test text for memory integration"
        
        result = execute(text, save_to_memory=True)
        
        assert result['success'] is True
        # Memory ID should be in metadata if save was successful
        if 'memory_id' in result['metadata']:
            assert isinstance(result['metadata']['memory_id'], int)
    
    @pytest.mark.asyncio
    async def test_pipeline_performance(self, sample_texts):
        """Test pipeline performance with multiple texts."""
        texts = [text for text in sample_texts if text.strip()][:5]
        
        import time
        start_time = time.time()
        
        results = batch_execute(texts, parallel=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 10.0  # 10 seconds max
        assert len(results) == len(texts)
        
        # All results should be successful (for valid texts)
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == len(texts)
