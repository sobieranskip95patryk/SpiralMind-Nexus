from typing import List, Dict, Any
import re

class RequestClassifier:
    """
    Klasyfikator żądań z możliwością rozszerzenia o nowe kategorie
    i zaawansowane heurystyki rozpoznawania
    """
    
    def __init__(self):
        # Podstawowe keywords dla różnych typów żądań
        self.keywords_creative = ['zaprojektuj', 'napisz', 'opowiedz', 'wyobraź', 'stwórz', 'wymyśl']
        self.keywords_analytical = ['przeanalizuj', 'sprawdź', 'zweryfikuj', 'oblicz', 'porównaj']
        self.keywords_technical = ['zaimplementuj', 'zakoduj', 'uruchom', 'debuguj', 'zoptymalizuj']
        self.keywords_x_platform = ['grok', 'xai', 'aigents', 'x platform', 'twitter', 'elon']
        
        # Wzorce regex dla zaawansowanego rozpoznawania
        self.creative_patterns = [
            r'\b(jak|co|czy)\s+(można|warto|lepiej)\s+\w+',
            r'\b(pomysł|idea|koncepcja|wizja)\b',
            r'\b(innowacyjn|kreatywn|oryginaln)\w*\b'
        ]
        
        self.analytical_patterns = [
            r'\b(analiza|badanie|raport|statystyki)\b',
            r'\b(dlaczego|jak|gdzie|kiedy|czemu)\s+\w+',
            r'\b(problem|błąd|issue|bug)\b'
        ]
    
    def is_creative_request(self, data: str) -> bool:
        """
        Sprawdza czy żądanie ma charakter kreatywny
        """
        data_lower = data.lower()
        
        # Sprawdź keywords
        if any(keyword in data_lower for keyword in self.keywords_creative):
            return True
        
        # Sprawdź wzorce regex
        for pattern in self.creative_patterns:
            if re.search(pattern, data_lower):
                return True
        
        return False
    
    def is_analytical_request(self, data: str) -> bool:
        """
        Sprawdza czy żądanie ma charakter analityczny
        """
        data_lower = data.lower()
        
        # Sprawdź keywords
        if any(keyword in data_lower for keyword in self.keywords_analytical):
            return True
        
        # Sprawdź wzorce regex
        for pattern in self.analytical_patterns:
            if re.search(pattern, data_lower):
                return True
        
        return False
    
    def is_technical_request(self, data: str) -> bool:
        """
        Sprawdza czy żądanie ma charakter techniczny
        """
        data_lower = data.lower()
        return any(keyword in data_lower for keyword in self.keywords_technical)
    
    def is_x_platform_related(self, data: str) -> bool:
        """
        Sprawdza czy żądanie dotyczy platformy X
        """
        data_lower = data.lower()
        return any(keyword in data_lower for keyword in self.keywords_x_platform)
    
    def classify_request_type(self, data: str) -> str:
        """
        Klasyfikuje typ żądania i zwraca główną kategorię
        """
        if self.is_x_platform_related(data):
            return 'X_PLATFORM'
        elif self.is_creative_request(data):
            return 'CREATIVE'
        elif self.is_analytical_request(data):
            return 'ANALYTICAL'
        elif self.is_technical_request(data):
            return 'TECHNICAL'
        else:
            return 'GENERAL'
    
    def get_request_features(self, data: str) -> Dict[str, Any]:
        """
        Zwraca szczegółowe cechy żądania dla analityki
        """
        return {
            'type': self.classify_request_type(data),
            'is_creative': self.is_creative_request(data),
            'is_analytical': self.is_analytical_request(data),
            'is_technical': self.is_technical_request(data),
            'is_x_platform': self.is_x_platform_related(data),
            'length': len(data),
            'word_count': len(data.split()),
            'has_questions': '?' in data,
            'has_exclamations': '!' in data
        }
    
    def add_keywords(self, category: str, new_keywords: List[str]):
        """
        Dodaje nowe keywords do określonej kategorii
        """
        if category == 'creative':
            self.keywords_creative.extend(new_keywords)
        elif category == 'analytical':
            self.keywords_analytical.extend(new_keywords)
        elif category == 'technical':
            self.keywords_technical.extend(new_keywords)
        elif category == 'x_platform':
            self.keywords_x_platform.extend(new_keywords)
    
    def add_pattern(self, category: str, pattern: str):
        """
        Dodaje nowy wzorzec regex do określonej kategorii
        """
        if category == 'creative':
            self.creative_patterns.append(pattern)
        elif category == 'analytical':
            self.analytical_patterns.append(pattern)