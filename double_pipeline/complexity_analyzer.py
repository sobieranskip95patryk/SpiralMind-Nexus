import re
from typing import Dict, Any

class ComplexityAnalyzer:
    """
    Analizator złożoności tekstu z możliwością rozszerzenia
    o zaawansowane metryki językowe
    """
    
    def __init__(self):
        self.base_length_threshold = 500.0
        self.complexity_weights = {
            'length': 0.4,
            'sentences': 0.2,
            'words': 0.2,
            'punctuation': 0.1,
            'special_chars': 0.1
        }
    
    def calculate(self, data: str) -> float:
        """
        Podstawowa analiza złożoności oparta na długości tekstu
        """
        if not data:
            return 0.0
        
        return min(1.0, len(data) / self.base_length_threshold)
    
    def calculate_advanced(self, data: str, metadata: Dict[str, Any] = None) -> float:
        """
        Zaawansowana analiza złożoności z wieloma metrykami
        """
        if not data:
            return 0.0
        
        # Metryki bazowe
        length_complexity = min(1.0, len(data) / self.base_length_threshold)
        
        # Analiza struktury zdań
        sentences = self._count_sentences(data)
        sentence_complexity = min(1.0, sentences / 10.0)  # Normalizacja do 10 zdań
        
        # Analiza słów
        words = len(data.split())
        word_complexity = min(1.0, words / 100.0)  # Normalizacja do 100 słów
        
        # Analiza interpunkcji
        punctuation_count = len([c for c in data if c in '.,!?;:'])
        punctuation_complexity = min(1.0, punctuation_count / 20.0)
        
        # Analiza znaków specjalnych
        special_chars = len([c for c in data if not c.isalnum() and not c.isspace()])
        special_complexity = min(1.0, special_chars / 30.0)
        
        # Złożoność ważona
        total_complexity = (
            length_complexity * self.complexity_weights['length'] +
            sentence_complexity * self.complexity_weights['sentences'] +
            word_complexity * self.complexity_weights['words'] +
            punctuation_complexity * self.complexity_weights['punctuation'] +
            special_complexity * self.complexity_weights['special_chars']
        )
        
        # Boost dla X platform content
        if metadata and 'Post z X' in data:
            x_complexity_boost = metadata.get('complexity', 0.0)
            total_complexity = min(1.0, total_complexity + x_complexity_boost * 0.2)
        
        return total_complexity
    
    def _count_sentences(self, text: str) -> int:
        """
        Zlicza zdania w tekście
        """
        # Prosty regex dla zdań
        sentences = re.split(r'[.!?]+', text.strip())
        return len([s for s in sentences if s.strip()])
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Aktualizuje wagi dla różnych metryk złożoności
        """
        for key, value in new_weights.items():
            if key in self.complexity_weights:
                self.complexity_weights[key] = value
    
    def get_complexity_breakdown(self, data: str) -> Dict[str, float]:
        """
        Zwraca rozbicie złożoności na poszczególne komponenty
        """
        if not data:
            return {'length': 0.0, 'sentences': 0.0, 'words': 0.0, 'punctuation': 0.0, 'special_chars': 0.0}
        
        return {
            'length': min(1.0, len(data) / self.base_length_threshold),
            'sentences': min(1.0, self._count_sentences(data) / 10.0),
            'words': min(1.0, len(data.split()) / 100.0),
            'punctuation': min(1.0, len([c for c in data if c in '.,!?;:']) / 20.0),
            'special_chars': min(1.0, len([c for c in data if not c.isalnum() and not c.isspace()]) / 30.0)
        }