"""Synergy Orchestrator for SpiralMind-Nexus.

Orchestrates the interaction between quantum and GOKAI processing.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..core.quantum_core import QuantumCore, QuantumResult
from ..core.gokai_core import GOKAICalculator, Score
from ..config.loader import Cfg

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of synergy processing."""
    
    quantum_result: QuantumResult
    gokai_score: Score
    processing_time: float
    decision: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'quantum_result': {
                'fibonacci_score': self.quantum_result.fibonacci_score,
                'entropy_score': self.quantum_result.entropy_score,
                'complexity_score': self.quantum_result.complexity_score,
                's9_score': self.quantum_result.s9_score,
                'quantum_score': self.quantum_result.quantum_score,
                'metadata': self.quantum_result.metadata
            },
            'gokai_score': self.gokai_score.to_dict(),
            'processing_time': self.processing_time,
            'decision': self.decision,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class ProcessingRequest:
    """Processing request structure."""
    
    text: str
    context: Dict[str, Any]
    priority: int = 0
    timestamp: datetime = None
    request_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"req_{int(time.time() * 1000)}"


class SynergyOrchestrator:
    """Orchestrates quantum and GOKAI processing with decision logic."""
    
    def __init__(self, config: Cfg = None):
        """Initialize synergy orchestrator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Cfg()
        
        # Initialize processing components
        self.quantum_core = QuantumCore(self.config.quantum.__dict__)
        self.gokai_calculator = GOKAICalculator(self.config.gokai.__dict__)
        
        # Processing configuration
        self.pipeline_config = self.config.pipeline
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.pipeline_config.parallel_workers)
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'average_processing_time': 0.0,
            'decisions': {
                'accept': 0,
                'reject': 0,
                'review': 0,
                'unknown': 0
            }
        }
        
        # Cache for recent results
        self._result_cache: Dict[str, ProcessingResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Decision thresholds
        self.decision_thresholds = {
            'accept_quantum': 0.7,
            'accept_gokai': 0.6,
            'accept_confidence': 0.5,
            'reject_quantum': 0.3,
            'reject_gokai': 0.2,
            'reject_confidence': 0.8
        }
        
        logger.info("SynergyOrchestrator initialized")
    
    def _clean_cache(self) -> None:
        """Clean expired cache entries."""
        if not self.pipeline_config.enable_caching:
            return
            
        current_time = datetime.now()
        ttl = timedelta(seconds=self.pipeline_config.cache_ttl)
        
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > ttl
        ]
        
        for key in expired_keys:
            self._result_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _get_cache_key(self, text: str, context: Dict[str, Any]) -> str:
        """Generate cache key for text and context.
        
        Args:
            text: Input text
            context: Processing context
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a deterministic hash of text and context
        content = f"{text}:{sorted(context.items())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _make_decision(self, 
                      quantum_result: QuantumResult, 
                      gokai_score: Score,
                      context: Dict[str, Any]) -> tuple[str, float]:
        """Make processing decision based on scores.
        
        Args:
            quantum_result: Quantum processing result
            gokai_score: GOKAI score
            context: Processing context
            
        Returns:
            Tuple of (decision, confidence)
        """
        quantum_score = quantum_result.quantum_score
        gokai_value = gokai_score.value
        gokai_confidence = gokai_score.confidence
        
        # Decision logic based on thresholds
        if (quantum_score >= self.decision_thresholds['accept_quantum'] and 
            gokai_value >= self.decision_thresholds['accept_gokai'] and
            gokai_confidence >= self.decision_thresholds['accept_confidence']):
            decision = "accept"
            confidence = min(gokai_confidence, 0.9)
            
        elif (quantum_score <= self.decision_thresholds['reject_quantum'] or 
              gokai_value <= self.decision_thresholds['reject_gokai']):
            decision = "reject"
            confidence = max(gokai_confidence, 0.7) if gokai_confidence > self.decision_thresholds['reject_confidence'] else 0.3
            
        else:
            decision = "review"
            confidence = gokai_confidence * 0.8  # Lower confidence for review items
        
        # Consider context factors
        if 'importance' in context:
            importance = float(context['importance'])
            if importance > 0.8 and decision == "reject":
                decision = "review"
                confidence *= 0.9
            elif importance < 0.2 and decision == "accept":
                decision = "review"
                confidence *= 0.8
        
        # Consider urgency
        if 'urgency' in context:
            urgency = float(context['urgency'])
            if urgency > 0.8:
                confidence *= 1.1  # Higher confidence for urgent items
            elif urgency < 0.2:
                confidence *= 0.9  # Lower confidence for non-urgent items
        
        confidence = min(1.0, max(0.0, confidence))
        
        logger.debug(f"Decision: {decision} (confidence: {confidence:.3f}) - "
                    f"Quantum: {quantum_score:.3f}, GOKAI: {gokai_value:.3f}")
        
        return decision, confidence
    
    def process(self, 
               text: str, 
               context: Dict[str, Any] = None,
               use_cache: bool = True) -> ProcessingResult:
        """Process text through synergy pipeline.
        
        Args:
            text: Input text to process
            context: Processing context
            use_cache: Whether to use caching
            
        Returns:
            ProcessingResult with decision and scores
        """
        start_time = time.time()
        context = context or {}
        
        logger.debug(f"Processing text of length {len(text)}")
        
        try:
            # Check cache first
            cache_key = None
            if self.pipeline_config.enable_caching and use_cache:
                cache_key = self._get_cache_key(text, context)
                if cache_key in self._result_cache:
                    logger.debug("Cache hit for processing request")
                    return self._result_cache[cache_key]
            
            # Clean expired cache entries periodically
            if len(self._result_cache) % 100 == 0:
                self._clean_cache()
            
            # Process through quantum core
            quantum_result = self.quantum_core.process(text, context)
            
            # Process through GOKAI calculator
            gokai_score = self.gokai_calculator.calculate(
                quantum_result.quantum_score, text, context
            )
            
            # Make decision
            decision, confidence = self._make_decision(quantum_result, gokai_score, context)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                quantum_result=quantum_result,
                gokai_score=gokai_score,
                processing_time=processing_time,
                decision=decision,
                confidence=confidence,
                metadata={
                    'pipeline_mode': self.pipeline_config.mode,
                    'processing_timestamp': datetime.now().isoformat(),
                    'cache_used': False,
                    'context_provided': bool(context)
                }
            )
            
            # Update statistics
            self._update_stats(result, success=True)
            
            # Cache result if enabled
            if self.pipeline_config.enable_caching and cache_key:
                self._result_cache[cache_key] = result
                self._cache_timestamps[cache_key] = datetime.now()
            
            logger.debug(f"Processing complete in {processing_time:.3f}s: {decision}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in synergy processing: {e}")
            
            # Create error result
            error_result = ProcessingResult(
                quantum_result=QuantumResult(
                    fibonacci_score=0.0,
                    entropy_score=0.0,
                    complexity_score=0.0,
                    s9_score=0.0,
                    quantum_score=0.0,
                    metadata={'error': str(e)}
                ),
                gokai_score=Score(
                    value=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                ),
                processing_time=processing_time,
                decision="unknown",
                confidence=0.0,
                metadata={'error': str(e), 'processing_failed': True}
            )
            
            self._update_stats(error_result, success=False)
            return error_result
    
    async def process_async(self, 
                           text: str, 
                           context: Dict[str, Any] = None) -> ProcessingResult:
        """Process text asynchronously.
        
        Args:
            text: Input text to process
            context: Processing context
            
        Returns:
            ProcessingResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.process, text, context
        )
    
    def batch_process(self, 
                     requests: List[ProcessingRequest],
                     parallel: bool = True) -> List[ProcessingResult]:
        """Process multiple requests.
        
        Args:
            requests: List of processing requests
            parallel: Whether to process in parallel
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Batch processing {len(requests)} requests (parallel={parallel})")
        
        if not parallel or len(requests) <= 1:
            # Sequential processing
            results = []
            for request in requests:
                result = self.process(request.text, request.context)
                result.metadata['request_id'] = request.request_id
                results.append(result)
            return results
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.pipeline_config.parallel_workers) as executor:
            futures = []
            for request in requests:
                future = executor.submit(self.process, request.text, request.context)
                futures.append((future, request))
            
            results = []
            for future, request in futures:
                try:
                    result = future.result(timeout=self.pipeline_config.timeout)
                    result.metadata['request_id'] = request.request_id
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    # Add error result
                    error_result = ProcessingResult(
                        quantum_result=QuantumResult(
                            fibonacci_score=0.0,
                            entropy_score=0.0,
                            complexity_score=0.0,
                            s9_score=0.0,
                            quantum_score=0.0,
                            metadata={'error': str(e)}
                        ),
                        gokai_score=Score(
                            value=0.0,
                            confidence=0.0,
                            timestamp=datetime.now(),
                            metadata={'error': str(e)}
                        ),
                        processing_time=0.0,
                        decision="unknown",
                        confidence=0.0,
                        metadata={
                            'error': str(e), 
                            'request_id': request.request_id,
                            'processing_failed': True
                        }
                    )
                    results.append(error_result)
        
        return results
    
    async def batch_process_async(self, 
                                 requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process multiple requests asynchronously.
        
        Args:
            requests: List of processing requests
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Async batch processing {len(requests)} requests")
        
        tasks = []
        for request in requests:
            task = self.process_async(request.text, request.context)
            tasks.append((task, request))
        
        results = []
        for task, request in tasks:
            try:
                result = await task
                result.metadata['request_id'] = request.request_id
                results.append(result)
            except Exception as e:
                logger.error(f"Error in async processing for {request.request_id}: {e}")
                # Add error result
                error_result = ProcessingResult(
                    quantum_result=QuantumResult(
                        fibonacci_score=0.0,
                        entropy_score=0.0,
                        complexity_score=0.0,
                        s9_score=0.0,
                        quantum_score=0.0,
                        metadata={'error': str(e)}
                    ),
                    gokai_score=Score(
                        value=0.0,
                        confidence=0.0,
                        timestamp=datetime.now(),
                        metadata={'error': str(e)}
                    ),
                    processing_time=0.0,
                    decision="unknown",
                    confidence=0.0,
                    metadata={
                        'error': str(e), 
                        'request_id': request.request_id,
                        'processing_failed': True
                    }
                )
                results.append(error_result)
        
        return results
    
    def _update_stats(self, result: ProcessingResult, success: bool) -> None:
        """Update processing statistics.
        
        Args:
            result: Processing result
            success: Whether processing was successful
        """
        self.stats['total_processed'] += 1
        
        if success:
            self.stats['successful_processed'] += 1
            
            # Update average processing time
            current_avg = self.stats['average_processing_time']
            total_successful = self.stats['successful_processed']
            
            new_avg = ((current_avg * (total_successful - 1)) + result.processing_time) / total_successful
            self.stats['average_processing_time'] = new_avg
            
            # Update decision counts
            decision = result.decision
            if decision in self.stats['decisions']:
                self.stats['decisions'][decision] += 1
        else:
            self.stats['failed_processed'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Add derived statistics
        total = stats['total_processed']
        if total > 0:
            stats['success_rate'] = stats['successful_processed'] / total
            stats['failure_rate'] = stats['failed_processed'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add cache statistics
        stats['cache_size'] = len(self._result_cache)
        
        # Add component statistics
        stats['quantum_stats'] = getattr(self.quantum_core, 'stats', {})
        stats['gokai_stats'] = self.gokai_calculator.get_statistics()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        self._result_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Processing cache cleared")
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'average_processing_time': 0.0,
            'decisions': {
                'accept': 0,
                'reject': 0,
                'review': 0,
                'unknown': 0
            }
        }
        logger.info("Processing statistics reset")
    
    def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources."""
        logger.info("Shutting down SynergyOrchestrator")
        self.executor.shutdown(wait=True)
        self.clear_cache()
