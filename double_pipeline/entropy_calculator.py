import math
from typing import Dict, Any

class EntropyCalculator:
    """
    Modułowy kalkulator entropii Shannon'a z możliwością rozszerzenia
    o dodatkowe metryki złożoności tekstu
    """
    
    def __init__(self):
        self.cache = {}  # Cache dla często używanych obliczeń
    
    def calculate(self, data: str) -> float:
        """
        Oblicza entropię Shannon'a dla tekstu
        """
        if not data:
            return 0.0
            
        # Sprawdź cache
        if data in self.cache:
            return self.cache[data]
        
        # Podstawowa entropia Shannon'a
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        data_len = len(data)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Cache wynik
        self.cache[data] = entropy
        return entropy
    
    def calculate_x_enhanced(self, data: str, metadata: Dict[str, Any] = None) -> float:
        """
        Rozszerzona entropia dla X posts z uwzględnieniem metadanych
        """
        base_entropy = self.calculate(data)
        
        if not metadata:
            return base_entropy
        
        # Boost dla X platform content
        if 'Post z X' in data:
            base_entropy += 0.2
        
        # Media boost
        media_type = metadata.get('media_type', 'text')
        if media_type == 'image':
            base_entropy += 0.3
        elif media_type == 'video':
            base_entropy += 0.4
        
        # Entropy boost from event metadata
        base_entropy += metadata.get('entropy_boost', 0.0)
        
        return base_entropy
    
    def clear_cache(self):
        """Wyczyść cache entropii"""
        self.cache.clear()