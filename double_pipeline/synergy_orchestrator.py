from typing import Dict, Any, Callable, List
import logging
from .entropy_calculator import EntropyCalculator
from .complexity_analyzer import ComplexityAnalyzer
from .request_classifier import RequestClassifier
from .config import GOKAI_CONFIG

class SynergyOrchestrator:
    """
    Modułowy orkiestrator SYNERGY z rozszerzalnym drzewem decyzyjnym
    i zaawansowaną analityką dla platform X
    """
    
    def __init__(self, shared_state: Dict[str, Any]):
        self.shared_state = shared_state
        self.entropy_calc = EntropyCalculator()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.request_classifier = RequestClassifier()
        
        # Konfiguracja z rozszerzeniami
        self.config = GOKAI_CONFIG.get('synergy_config', {
            'confidence_threshold': 0.75,
            'success_threshold': 85.0,
            'complexity_high': 0.8,
            'entropy_high': 2.5,
            'x_platform_boost': 0.2,
            'leadership_penalty': 0.15,
            'video_creativity_boost': 0.3
        })
        
        # Rozszerzalne reguły decyzyjne
        self.custom_rules: List[Callable] = []
        
        # Statystyki dla analityki
        self.decision_stats = {
            'verification_count': 0,
            'creative_count': 0,
            'balanced_count': 0,
            'standard_count': 0,
            'x_platform_count': 0
        }
    
    def orchestrate(self, current_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Główne drzewo decyzyjne z rozszerzeniami dla X platform
        """
        payload = current_event.get('payload', '')
        
        # Użyj rozszerzonych kalkulatorów
        entropy = self.entropy_calc.calculate_x_enhanced(payload, current_event)
        complexity = self.complexity_analyzer.calculate_advanced(payload, current_event)
        
        # Klasyfikacja żądania
        request_features = self.request_classifier.get_request_features(payload)
        is_creative = request_features['is_creative']
        is_x_platform = request_features['is_x_platform']
        
        # Stan systemu
        last_confidence = self.shared_state.get('last_confidence', 1.0)
        last_success_pct = self.shared_state.get('last_success_pct', 100.0)
        self.shared_state['current_event_payload'] = payload
        
        # ===== ROZSZERZENIA DLA X PLATFORM =====
        if is_x_platform or 'Post z X' in payload:
            self.decision_stats['x_platform_count'] += 1
            return self._handle_x_platform_logic(current_event, entropy, complexity, is_creative, last_confidence, last_success_pct)
        
        # ===== PODSTAWOWE DRZEWO DECYZYJNE =====
        decision = self._apply_base_logic(entropy, complexity, is_creative, last_confidence, last_success_pct)
        
        # ===== ROZSZERZALNE REGUŁY =====
        for custom_rule in self.custom_rules:
            decision = custom_rule(decision, current_event, self.shared_state)
        
        # Aktualizuj statystyki
        self._update_decision_stats(decision['mode'])
        
        return decision
    
    def _handle_x_platform_logic(self, event: Dict[str, Any], entropy: float, complexity: float, is_creative: bool, last_confidence: float, last_success_pct: float) -> Dict[str, Any]:
        """
        Specjalna logika dla X platform posts z zaawansowanymi heurystykami
        """
        payload = event.get('payload', '')
        
        # X platform boost
        complexity += self.config['x_platform_boost']
        
        # Advanced AI keywords detection
        if any(keyword in payload.lower() for keyword in ['grok', 'xai', 'aigents']):
            is_creative = True
            complexity += 0.1
            post_id = payload[8:11] if len(payload) > 10 else 'X'
            logging.info(f"SYNERGY X-Mode: Wykryto advanced AI keywords w post ID {post_id}")
        
        # Video/Media enhanced logic
        if event.get('media_type') == 'video':
            complexity += self.config['video_creativity_boost']
            is_creative = True
            logging.info("SYNERGY X-Mode: Video content detected - switching to CREATIVE mode")
        elif event.get('media_type') == 'image':
            complexity += 0.15
            entropy += 0.1
        
        # Leadership content special handling
        if any(leader in payload.lower() for leader in ['ceo', 'yaccarino', 'musk']):
            last_confidence *= (1.0 - self.config['leadership_penalty'])
            logging.info("SYNERGY X-Mode: Leadership content - enhanced verification needed")
        
        # Algorithmiczne podejście dla X
        if last_confidence < 0.7 or last_success_pct < 80.0:
            logging.info("SYNERGY X-Mode: Tryb Weryfikacji (X-VERIFICATION).")
            return {'pipelines': ['LOGIKA:AI'], 'mode': 'X_VERIFICATION_FOCUS', 'logic_bias': 0.95, 'x_enhanced': True}
        elif complexity > 0.7 and entropy > 2.0:
            logging.info("SYNERGY X-Mode: Tryb Równoległy (X-PARALLEL).")
            return {'pipelines': ['GOK:AI', 'LOGIKA:AI'], 'mode': 'X_BALANCED_BLEND', 'alpha': 0.6, 'x_enhanced': True}
        elif is_creative:
            logging.info("SYNERGY X-Mode: Tryb Kreatywny (X-CREATIVE).")
            return {'pipelines': ['GOK:AI'], 'mode': 'X_CREATIVE_EXPLORATION', 'alpha': 0.95, 'x_enhanced': True}
        else:
            logging.info("SYNERGY X-Mode: Tryb Standardowy (X-STANDARD).")
            return {'pipelines': ['GOK:AI'], 'mode': 'X_STANDARD', 'x_enhanced': True}
    
    def _apply_base_logic(self, entropy: float, complexity: float, is_creative: bool, last_confidence: float, last_success_pct: float) -> Dict[str, Any]:
        """
        Podstawowa logika decyzyjna (bez X platform)
        """
        if last_confidence < self.config['confidence_threshold'] or last_success_pct < self.config['success_threshold']:
            logging.info("SYNERGY: Tryb Weryfikacji (VERIFICATION).")
            return {'pipelines': ['LOGIKA:AI'], 'mode': 'VERIFICATION_FOCUS', 'logic_bias': 0.9}
        elif complexity > self.config['complexity_high'] and entropy > self.config['entropy_high']:
            logging.info("SYNERGY: Tryb Równoległy (PARALLEL_PROCESSING).")
            return {'pipelines': ['GOK:AI', 'LOGIKA:AI'], 'mode': 'BALANCED_BLEND', 'alpha': 0.5}
        elif is_creative:
            logging.info("SYNERGY: Tryb Kreatywny (CREATIVE_EXPLORATION).")
            return {'pipelines': ['GOK:AI'], 'mode': 'CREATIVE_EXPLORATION', 'alpha': 0.9}
        else:
            logging.info("SYNERGY: Tryb Standardowy (STANDARD_OPERATION).")
            return {'pipelines': ['GOK:AI'], 'mode': 'STANDARD'}
    
    def _update_decision_stats(self, mode: str):
        """
        Aktualizuje statystyki decyzji dla analityki
        """
        if 'VERIFICATION' in mode:
            self.decision_stats['verification_count'] += 1
        elif 'CREATIVE' in mode:
            self.decision_stats['creative_count'] += 1
        elif 'BALANCED' in mode or 'BLEND' in mode:
            self.decision_stats['balanced_count'] += 1
        else:
            self.decision_stats['standard_count'] += 1
    
    def extend_decision_tree(self, custom_rule: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
        """
        Dodaje niestandardową regułę do drzewa decyzyjnego
        
        Args:
            custom_rule: funkcja(decision, event, shared_state) -> modified_decision
        """
        self.custom_rules.append(custom_rule)
        logging.info(f"SYNERGY: Dodano niestandardową regułę. Łącznie reguł: {len(self.custom_rules)}")
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Zwraca analitykę decyzji SYNERGY
        """
        total_decisions = sum(self.decision_stats.values())
        if total_decisions == 0:
            return self.decision_stats
        
        return {
            **self.decision_stats,
            'total_decisions': total_decisions,
            'verification_rate': self.decision_stats['verification_count'] / total_decisions,
            'creative_rate': self.decision_stats['creative_count'] / total_decisions,
            'x_platform_rate': self.decision_stats['x_platform_count'] / total_decisions,
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Aktualizuje konfigurację SYNERGY w runtime
        """
        self.config.update(new_config)
        logging.info(f"SYNERGY: Zaktualizowano konfigurację: {new_config}")
    
    def reset_stats(self):
        """
        Resetuje statystyki decyzji
        """
        for key in self.decision_stats:
            self.decision_stats[key] = 0
        logging.info("SYNERGY: Zresetowano statystyki decyzji")