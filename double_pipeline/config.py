from dataclasses import dataclass

@dataclass
class BaseParams:
    W: int = 7
    M: int = 6
    D: int = 4
    C: int = 5
    A: int = 8
    E: int = 6
    T: int = 3

GOKAI_CONFIG = {
    'base_params': BaseParams().__dict__,
    'MATRIX_347743': [3, 4, 7, 7, 4, 3],
    'alpha_schedule': [0.10, 0.20, 0.35, 0.50, 0.35, 0.20, 0.10],
    'max_fibonacci_n': 55,
    'personality': {
        'energy': {'extravert': 40, 'introvert': 60},
        'mind': {'intuitive': 67, 'observant': 33},
        'nature': {'thinking': 54, 'feeling': 46},
        'tactics': {'judging': 42, 'prospecting': 58},
        'identity': {'assertive': 43, 'turbulent': 57}
    },
    'logik_traits': {
        'autentyczna_wiez': 'Szczerość w relacjach, unikanie nieporozumień',
        'konflikt': 'Ignorowanie emocji, logika nad uczuciami',
        'logiczna_analiza': 'Systematyczne podejście do problemów',
        'precyzja': 'Dokładność w słowach i działaniach',
        'niezależność': 'Autonomia w myśleniu i działaniu',
        'kreatywność': 'Innowacyjne rozwiązania problemów',
        'ciągła_nauka': 'Stałe poszerzanie wiedzy',
        'strategiczne_myślenie': 'Długoterminowe planowanie',
        'kompetencja': 'Dążenie do mistrzostwa',
        'wyzwania': 'Poszukiwanie nowych problemów do rozwiązania',
        'efektywność': 'Optymalizacja procesów',
        'wizja_przyszłości': 'Przewidywanie trendów',
        'abstrakcyjne_myślenie': 'Teoretyczne podejście',
        'kwestiowanie': 'Podważanie status quo',
        'doskonaleń': 'Dążenie do idealu',
        'systemowe_spojrzenie': 'Holistyczne rozumienie'
    },
    'modes': {'enable_intp_mode': True, 'enable_einstein_mode': True, 'enable_gates_mode': True},
    
    # ===== NOWA KONFIGURACJA SYNERGY =====
    'synergy_config': {
        # Podstawowe progi decyzyjne
        'confidence_threshold': 0.75,
        'success_threshold': 85.0,
        'complexity_high': 0.8,
        'entropy_high': 2.5,
        
        # X Platform rozszerzenia
        'x_platform_boost': 0.2,
        'leadership_penalty': 0.15,
        'video_creativity_boost': 0.3,
        'image_entropy_boost': 0.1,
        
        # Zaawansowane heurystyki
        'grok_keyword_boost': 0.1,
        'ai_keyword_multiplier': 1.05,
        'media_complexity_multiplier': 1.2,
        
        # Progi dla X tryb'ów
        'x_verification_confidence': 0.7,
        'x_verification_success': 80.0,
        'x_parallel_complexity': 0.7,
        'x_parallel_entropy': 2.0,
        
        # Analityka i optymalizacja
        'enable_decision_analytics': True,
        'cache_entropy_calculations': True,
        'log_detailed_decisions': True,
        
        # Eksperymentalne features
        'experimental_x_modes': True,
        'adaptive_thresholds': False,
        'ml_enhanced_classification': False
    }
}
