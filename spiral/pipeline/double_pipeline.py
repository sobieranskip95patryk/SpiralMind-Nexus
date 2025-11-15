"""Double Pipeline for SpiralMind-Nexus.

Provides high-level pipeline functions for text processing and event creation.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid

from .synergy_orchestrator import SynergyOrchestrator, ProcessingRequest
from ..config.loader import Cfg
from ..memory.persistence import MemoryPersistence

logger = logging.getLogger(__name__)

# Global orchestrator instance
_orchestrator: SynergyOrchestrator = None
_memory_persistence: MemoryPersistence = None


def _get_orchestrator() -> SynergyOrchestrator:
    """Get or create global orchestrator instance.
    
    Returns:
        SynergyOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        config = Cfg()
        _orchestrator = SynergyOrchestrator(config)
        logger.info("Created global SynergyOrchestrator instance")
    return _orchestrator


def _get_memory_persistence() -> MemoryPersistence:
    """Get or create global memory persistence instance.
    
    Returns:
        MemoryPersistence instance
    """
    global _memory_persistence
    if _memory_persistence is None:
        _memory_persistence = MemoryPersistence()
        logger.info("Created global MemoryPersistence instance")
    return _memory_persistence


def execute(text: str, 
           context: Dict[str, Any] = None,
           mode: str = "quantum",
           save_to_memory: bool = False) -> Dict[str, Any]:
    """Execute text processing through the double pipeline.
    
    This is the main entry point for processing text through the SpiralMind-Nexus system.
    
    Args:
        text: Input text to process
        context: Additional context for processing
        mode: Processing mode ('quantum', 'gokai', 'hybrid')
        save_to_memory: Whether to save result to memory
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Executing pipeline for text (length: {len(text)}, mode: {mode})")
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text input provided")
        return {
            'success': False,
            'error': 'Invalid text input',
            'result': None
        }
    
    context = context or {}
    context['mode'] = mode
    context['pipeline_execution'] = True
    
    try:
        # Get orchestrator and process
        orchestrator = _get_orchestrator()
        
        # Create processing request
        request = ProcessingRequest(
            text=text,
            context=context,
            timestamp=datetime.now()
        )
        
        # Process through pipeline
        result = orchestrator.process(text, context)
        
        # Prepare response
        response = {
            'success': True,
            'result': result.to_dict(),
            'decision': result.decision,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'quantum_score': result.quantum_result.quantum_score,
            'gokai_score': result.gokai_score.value,
            'metadata': {
                'mode': mode,
                'request_id': request.request_id,
                'timestamp': request.timestamp.isoformat(),
                'text_length': len(text),
                'context_provided': bool(context)
            }
        }
        
        # Save to memory if requested
        if save_to_memory:
            try:
                memory = _get_memory_persistence()
                memory_entry = {
                    'text': text,
                    'context': context,
                    'result': result.to_dict(),
                    'timestamp': datetime.now().isoformat(),
                    'mode': mode
                }
                memory_id = memory.save_memory(memory_entry)
                response['metadata']['memory_id'] = memory_id
                logger.debug(f"Saved result to memory with ID: {memory_id}")
            except Exception as e:
                logger.warning(f"Failed to save to memory: {e}")
                response['metadata']['memory_save_error'] = str(e)
        
        logger.info(f"Pipeline execution successful: {result.decision} (confidence: {result.confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        return {
            'success': False,
            'error': str(e),
            'result': None,
            'metadata': {
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text)
            }
        }


def batch_execute(texts: List[str],
                 contexts: List[Dict[str, Any]] = None,
                 mode: str = "quantum",
                 parallel: bool = True,
                 save_to_memory: bool = False) -> List[Dict[str, Any]]:
    """Execute batch processing through the double pipeline.
    
    Args:
        texts: List of input texts to process
        contexts: Optional list of contexts (one per text)
        mode: Processing mode
        parallel: Whether to process in parallel
        save_to_memory: Whether to save results to memory
        
    Returns:
        List of processing results
    """
    logger.info(f"Executing batch pipeline for {len(texts)} texts (mode: {mode}, parallel: {parallel})")
    
    if not texts or not all(isinstance(text, str) for text in texts):
        logger.warning("Invalid texts input provided")
        return [{
            'success': False,
            'error': 'Invalid texts input',
            'result': None
        }] * len(texts)
    
    # Prepare contexts
    if contexts is None:
        contexts = [{}] * len(texts)
    elif len(contexts) != len(texts):
        # Extend or truncate contexts to match texts length
        if len(contexts) < len(texts):
            contexts.extend([{}] * (len(texts) - len(contexts)))
        else:
            contexts = contexts[:len(texts)]
    
    # Add mode to all contexts
    for context in contexts:
        context['mode'] = mode
        context['pipeline_execution'] = True
        context['batch_processing'] = True
    
    try:
        orchestrator = _get_orchestrator()
        
        # Create processing requests
        requests = [
            ProcessingRequest(text=text, context=context, timestamp=datetime.now())
            for text, context in zip(texts, contexts)
        ]
        
        # Process batch
        results = orchestrator.batch_process(requests, parallel=parallel)
        
        # Prepare responses
        responses = []
        for i, (result, request) in enumerate(zip(results, requests)):
            response = {
                'success': True,
                'result': result.to_dict(),
                'decision': result.decision,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'quantum_score': result.quantum_result.quantum_score,
                'gokai_score': result.gokai_score.value,
                'metadata': {
                    'mode': mode,
                    'request_id': request.request_id,
                    'batch_index': i,
                    'timestamp': request.timestamp.isoformat(),
                    'text_length': len(texts[i]),
                    'parallel_processing': parallel
                }
            }
            
            # Handle processing errors
            if 'error' in result.metadata:
                response['success'] = False
                response['error'] = result.metadata['error']
            
            responses.append(response)
        
        # Save to memory if requested
        if save_to_memory:
            memory = _get_memory_persistence()
            for i, (response, text, context) in enumerate(zip(responses, texts, contexts)):
                if response['success']:
                    try:
                        memory_entry = {
                            'text': text,
                            'context': context,
                            'result': response['result'],
                            'timestamp': datetime.now().isoformat(),
                            'mode': mode,
                            'batch_index': i
                        }
                        memory_id = memory.save_memory(memory_entry)
                        response['metadata']['memory_id'] = memory_id
                    except Exception as e:
                        logger.warning(f"Failed to save batch item {i} to memory: {e}")
                        response['metadata']['memory_save_error'] = str(e)
        
        successful_count = sum(1 for r in responses if r['success'])
        logger.info(f"Batch pipeline execution completed: {successful_count}/{len(responses)} successful")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch pipeline execution: {e}")
        return [{
            'success': False,
            'error': str(e),
            'result': None,
            'metadata': {
                'mode': mode,
                'batch_index': i,
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text) if i < len(texts) else 0
            }
        } for i, text in enumerate(texts)]


def create_event(text: str,
                event_type: str = "processing",
                metadata: Dict[str, Any] = None,
                process_immediately: bool = True) -> Dict[str, Any]:
    """Create and optionally process an event.
    
    Args:
        text: Event text content
        event_type: Type of event ('processing', 'analysis', 'decision', etc.)
        metadata: Additional event metadata
        process_immediately: Whether to process the event immediately
        
    Returns:
        Event creation and processing result
    """
    logger.info(f"Creating event: {event_type} (process: {process_immediately})")
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text for event creation")
        return {
            'success': False,
            'error': 'Invalid text input',
            'event': None
        }
    
    # Create event structure
    event_id = str(uuid.uuid4())
    event = {
        'id': event_id,
        'type': event_type,
        'text': text,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'processed': False,
        'result': None
    }
    
    try:
        # Save event to memory
        memory = _get_memory_persistence()
        memory_id = memory.save_memory({
            'event': event,
            'type': 'event_created',
            'timestamp': event['timestamp']
        })
        
        event['memory_id'] = memory_id
        
        # Process immediately if requested
        if process_immediately:
            processing_context = {
                'event_id': event_id,
                'event_type': event_type,
                'event_metadata': event['metadata']
            }
            
            # Execute processing
            processing_result = execute(
                text=text,
                context=processing_context,
                mode="quantum",
                save_to_memory=True
            )
            
            if processing_result['success']:
                event['processed'] = True
                event['result'] = processing_result['result']
                
                # Update event in memory
                memory.save_memory({
                    'event': event,
                    'type': 'event_processed',
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Event {event_id} created and processed successfully")
            else:
                logger.warning(f"Event {event_id} created but processing failed: {processing_result.get('error')}")
        
        return {
            'success': True,
            'event': event,
            'processed': event['processed'],
            'memory_id': memory_id
        }
        
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return {
            'success': False,
            'error': str(e),
            'event': event  # Return partial event even on error
        }


def get_pipeline_statistics() -> Dict[str, Any]:
    """Get comprehensive pipeline statistics.
    
    Returns:
        Statistics dictionary
    """
    try:
        orchestrator = _get_orchestrator()
        stats = orchestrator.get_statistics()
        
        # Add memory statistics if available
        try:
            memory = _get_memory_persistence()
            memory_stats = memory.get_statistics()
            stats['memory'] = memory_stats
        except Exception as e:
            logger.warning(f"Could not get memory statistics: {e}")
            stats['memory'] = {'error': str(e)}
        
        return {
            'success': True,
            'statistics': stats
        }
        
    except Exception as e:
        logger.error(f"Error getting pipeline statistics: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def reset_pipeline() -> Dict[str, Any]:
    """Reset pipeline state and statistics.
    
    Returns:
        Reset operation result
    """
    try:
        global _orchestrator, _memory_persistence
        
        if _orchestrator:
            _orchestrator.reset_statistics()
            _orchestrator.clear_cache()
            logger.info("Pipeline statistics and cache reset")
        
        return {
            'success': True,
            'message': 'Pipeline reset successfully'
        }
        
    except Exception as e:
        logger.error(f"Error resetting pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def shutdown_pipeline() -> Dict[str, Any]:
    """Shutdown pipeline and cleanup resources.
    
    Returns:
        Shutdown operation result
    """
    try:
        global _orchestrator, _memory_persistence
        
        if _orchestrator:
            _orchestrator.shutdown()
            _orchestrator = None
            
        if _memory_persistence:
            _memory_persistence.close()
            _memory_persistence = None
        
        logger.info("Pipeline shutdown completed")
        
        return {
            'success': True,
            'message': 'Pipeline shutdown successfully'
        }
        
    except Exception as e:
        logger.error(f"Error shutting down pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
        }
