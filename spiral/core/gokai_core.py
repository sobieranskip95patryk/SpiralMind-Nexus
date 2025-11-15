"""GOKAI Core module for SpiralMind-Nexus.

Implements the GOKAI scoring system and calculator.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Score:
    """Represents a GOKAI score with metadata."""
    
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    components: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate score after initialization."""
        self.value = max(0.0, min(1.0, self.value))
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        if self.components is None:
            self.components = {}
    
    @property
    def weighted_score(self) -> float:
        """Calculate confidence-weighted score."""
        return self.value * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert score to dictionary."""
        return {
            'value': self.value,
            'confidence': self.confidence,
            'weighted_score': self.weighted_score,
            'timestamp': self.timestamp.isoformat(),
            'components': self.components,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        return f"Score(value={self.value:.3f}, confidence={self.confidence:.3f})"


class GOKAICalculator:
    """GOKAI scoring calculator with advanced algorithms."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GOKAI calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default weights for score components
        self.weights = self.config.get('weights', {
            'quantum': 0.4,
            'semantic': 0.3,
            'contextual': 0.2,
            'temporal': 0.1
        })
        
        # Confidence calculation parameters
        self.confidence_params = self.config.get('confidence', {
            'base_confidence': 0.7,
            'length_factor': 0.1,
            'complexity_factor': 0.15,
            'consistency_factor': 0.05
        })
        
        # Score history for trend analysis
        self.score_history: List[Score] = []
        
    def calculate_base_score(self, 
                           quantum_score: float,
                           text: str = "",
                           context: Dict[str, Any] = None) -> float:
        """Calculate base GOKAI score.
        
        Args:
            quantum_score: Input quantum score
            text: Original text for additional analysis
            context: Additional context information
            
        Returns:
            Base GOKAI score
        """
        context = context or {}
        
        # Start with quantum score
        base = quantum_score * self.weights['quantum']
        
        # Add semantic component
        semantic_score = self._calculate_semantic_score(text, context)
        base += semantic_score * self.weights['semantic']
        
        # Add contextual component
        contextual_score = self._calculate_contextual_score(context)
        base += contextual_score * self.weights['contextual']
        
        # Add temporal component
        temporal_score = self._calculate_temporal_score(context)
        base += temporal_score * self.weights['temporal']
        
        return min(1.0, max(0.0, base))
    
    def _calculate_semantic_score(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate semantic component of score."""
        if not text:
            return 0.5  # Neutral score for empty text
            
        # Analyze text semantics
        words = text.lower().split()
        
        # Positive and negative word indicators
        positive_indicators = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                             'perfect', 'brilliant', 'outstanding', 'fantastic'}
        negative_indicators = {'bad', 'terrible', 'awful', 'horrible', 'worst',
                             'fail', 'failure', 'disaster', 'catastrophe'}
        
        positive_count = sum(1 for word in words if word in positive_indicators)
        negative_count = sum(1 for word in words if word in negative_indicators)
        
        # Calculate sentiment balance
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            sentiment = 0.5  # Neutral
        else:
            sentiment = positive_count / total_indicators
            
        # Consider text length and complexity
        length_factor = min(len(text) / 1000, 1.0) * 0.3
        complexity_factor = len(set(words)) / len(words) if words else 0.5
        
        semantic_score = sentiment * 0.6 + length_factor + complexity_factor * 0.1
        
        return min(1.0, semantic_score)
    
    def _calculate_contextual_score(self, context: Dict[str, Any]) -> float:
        """Calculate contextual component of score."""
        if not context:
            return 0.5
            
        score = 0.5  # Base contextual score
        
        # Factor in context richness
        context_richness = len(context) / 10.0  # Normalize by expected context size
        score += context_richness * 0.2
        
        # Factor in specific context indicators
        if 'importance' in context:
            importance = float(context.get('importance', 0.5))
            score += (importance - 0.5) * 0.3
            
        if 'urgency' in context:
            urgency = float(context.get('urgency', 0.5))
            score += (urgency - 0.5) * 0.2
            
        if 'quality' in context:
            quality = float(context.get('quality', 0.5))
            score += (quality - 0.5) * 0.3
            
        return min(1.0, max(0.0, score))
    
    def _calculate_temporal_score(self, context: Dict[str, Any]) -> float:
        """Calculate temporal component of score."""
        # Base temporal score
        base_temporal = 0.5
        
        # Consider recency if timestamp is available
        if 'timestamp' in context:
            try:
                timestamp = context['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                now = datetime.now()
                age_hours = (now - timestamp).total_seconds() / 3600
                
                # Recent content gets higher temporal score
                if age_hours < 1:
                    recency_factor = 0.3
                elif age_hours < 24:
                    recency_factor = 0.2
                elif age_hours < 168:  # 1 week
                    recency_factor = 0.1
                else:
                    recency_factor = 0.0
                    
                base_temporal += recency_factor
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing timestamp: {e}")
        
        # Consider seasonal or periodic factors
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            base_temporal += 0.1
            
        return min(1.0, base_temporal)
    
    def calculate_confidence(self, 
                           score: float,
                           text: str = "",
                           context: Dict[str, Any] = None) -> float:
        """Calculate confidence in the score.
        
        Args:
            score: Calculated score
            text: Original text
            context: Context information
            
        Returns:
            Confidence value (0.0 to 1.0)
        """
        context = context or {}
        
        # Start with base confidence
        confidence = self.confidence_params['base_confidence']
        
        # Adjust based on text length
        if text:
            length_factor = min(len(text) / 500, 1.0)  # Longer text = higher confidence
            confidence += length_factor * self.confidence_params['length_factor']
        
        # Adjust based on context richness
        context_factor = min(len(context) / 5, 1.0)
        confidence += context_factor * self.confidence_params['complexity_factor']
        
        # Adjust based on score consistency with history
        if self.score_history:
            recent_scores = [s.value for s in self.score_history[-5:]]  # Last 5 scores
            if recent_scores:
                avg_recent = sum(recent_scores) / len(recent_scores)
                consistency = 1.0 - abs(score - avg_recent)
                confidence += consistency * self.confidence_params['consistency_factor']
        
        return min(1.0, max(0.1, confidence))  # Ensure confidence is between 0.1 and 1.0
    
    def calculate(self, 
                 quantum_score: float,
                 text: str = "",
                 context: Dict[str, Any] = None) -> Score:
        """Calculate complete GOKAI score.
        
        Args:
            quantum_score: Input quantum score
            text: Original text
            context: Additional context
            
        Returns:
            Complete Score object
        """
        logger.debug(f"Calculating GOKAI score for quantum_score={quantum_score:.3f}")
        
        context = context or {}
        
        # Calculate base score
        base_score = self.calculate_base_score(quantum_score, text, context)
        
        # Calculate confidence
        confidence = self.calculate_confidence(base_score, text, context)
        
        # Create score components for transparency
        components = {
            'quantum': quantum_score * self.weights['quantum'],
            'semantic': self._calculate_semantic_score(text, context) * self.weights['semantic'],
            'contextual': self._calculate_contextual_score(context) * self.weights['contextual'],
            'temporal': self._calculate_temporal_score(context) * self.weights['temporal']
        }
        
        # Create final score
        score = Score(
            value=base_score,
            confidence=confidence,
            timestamp=datetime.now(),
            components=components,
            metadata={
                'quantum_input': quantum_score,
                'text_length': len(text),
                'context_keys': list(context.keys()),
                'weights_used': self.weights.copy()
            }
        )
        
        # Add to history
        self.score_history.append(score)
        
        # Limit history size
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-50:]
            
        logger.debug(f"GOKAI calculation complete: {score}")
        return score
    
    def batch_calculate(self, 
                       quantum_scores: List[float],
                       texts: List[str] = None,
                       contexts: List[Dict[str, Any]] = None) -> List[Score]:
        """Calculate scores for multiple inputs.
        
        Args:
            quantum_scores: List of quantum scores
            texts: Optional list of texts
            contexts: Optional list of contexts
            
        Returns:
            List of Score objects
        """
        logger.info(f"Batch calculating {len(quantum_scores)} GOKAI scores")
        
        texts = texts or [""] * len(quantum_scores)
        contexts = contexts or [{}] * len(quantum_scores)
        
        # Ensure all lists have the same length
        max_len = max(len(quantum_scores), len(texts), len(contexts))
        texts.extend([""] * (max_len - len(texts)))
        contexts.extend([{}] * (max_len - len(contexts)))
        
        scores = []
        for i, (qs, text, context) in enumerate(zip(quantum_scores, texts, contexts)):
            try:
                score = self.calculate(qs, text, context)
                score.metadata['batch_index'] = i
                scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating score {i}: {e}")
                # Create error score
                error_score = Score(
                    value=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'batch_index': i, 'error': str(e)}
                )
                scores.append(error_score)
                
        return scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about score history.
        
        Returns:
            Dictionary with statistics
        """
        if not self.score_history:
            return {'count': 0}
            
        scores = [s.value for s in self.score_history]
        confidences = [s.confidence for s in self.score_history]
        
        stats = {
            'count': len(scores),
            'score_stats': {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': math.sqrt(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))
            },
            'confidence_stats': {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'latest_score': self.score_history[-1].to_dict()
        }
        
        return stats
