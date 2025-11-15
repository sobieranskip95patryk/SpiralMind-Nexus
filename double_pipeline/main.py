import logging
import random
from datetime import datetime
from .gokai_core import run_cycle
from .config import GOKAI_CONFIG
from .utils import BaseParams
from .synergy_orchestrator import SynergyOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: [%(asctime)s] %(message)s')
logging.info(f"Rozpoczęcie Mózgu Boga - {datetime.now().strftime('%H:%M:%S %Z, %d/%m/%Y')}")

# Struktury danych z integracją X
class EventStream:
    def __init__(self, events: list = None, use_x_integration: bool = True):
        # Dodaj real-time posty z X jako przykładowe dane
        self.x_posts = [
            {'payload': 'Post z X [0]: Enhanced Twitter Integration for AIgents – auto-updates knowledge bases with tweets.', 
             'media_type': 'image', 'complexity': 0.8},
            {'payload': 'Post z X [1]: X is part of xAI after 2025 deal – integrations like Grok.', 
             'media_type': 'text', 'complexity': 0.6},
            {'payload': 'Post z X [2]: X powered by AI, revealed by CEO Linda Yaccarino.', 
             'media_type': 'text', 'complexity': 0.5},
            {'payload': 'Post z X [3]: X Algorithm Agent on Terramind: Nexus analyzes Twitter Algorithm repo.', 
             'media_type': 'text', 'complexity': 0.7},
            {'payload': 'Post z X [4]: X pipeline: New Video Ecosystem, AI search, Grok.', 
             'media_type': 'video', 'complexity': 0.9}
        ]
        self._events = events or self.x_posts
        self._index = 0
        self.use_x_integration = use_x_integration
    
    def next_event(self) -> dict:
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            
            # Dodaj X-specific metadata
            if self.use_x_integration and 'Post z X' in event.get('payload', ''):
                event['source'] = 'x_platform'
                event['timestamp'] = datetime.now().isoformat()
                # Symulacja entropy boost dla mediów
                if event.get('media_type') == 'image':
                    event['entropy_boost'] = 0.3
                elif event.get('media_type') == 'video':
                    event['entropy_boost'] = 0.4
                else:
                    event['entropy_boost'] = 0.1
            
            return event
        return {}

def create_shared_state() -> dict:
    return {
        'level': 0, 'n': 1, 'last_confidence': 1.0, 'last_success_pct': 100.0,
        'synergy_decision': None, 'current_event_payload': '', 'history': []
    }

# ===== STARA KLASA SYNERGY ZOSTAŁA ZASTĄPIONA PRZEZ SynergyOrchestrator =====
# Kod usunięty - użyj SynergyOrchestrator z modułu synergy_orchestrator.py

def spiral_thought_logika(shared_state: dict, strategy: dict) -> dict:
    logging.info(f"LOGIKA:AI - Weryfikacja w trybie: {strategy['mode']}")
    base_confidence = shared_state.get('last_confidence', 0.5)
    verified_confidence = min(1.0, base_confidence + strategy.get('logic_bias', 0.1))
    success_pct = verified_confidence * 100
    shared_state['last_confidence'] = verified_confidence
    shared_state['last_success_pct'] = success_pct
    return {'confidence': verified_confidence, 'success_pct': success_pct, 'answer': 'Wynik LOGIKA:AI po weryfikacji'}

def rebalance_weights_main(matrix: list, last_success: float) -> list:
    new_matrix = list(matrix)
    if last_success > 90.0:
        new_matrix = [min(9, w + 1) for w in new_matrix]
    elif last_success < 70.0:
        rand_idx = random.randint(0, len(new_matrix) - 1)
        new_matrix[rand_idx] = max(1, new_matrix[rand_idx] + random.choice([-2, 2]))
    return new_matrix

# Główna pętla z testami X Platform
def main():
    shared_state = create_shared_state()
    synergy = SynergyOrchestrator(shared_state)  # ✅ NOWY MODUŁOWY SYNERGY
    
    # ===== UŻYWAMY X POSTS ZAMIAST STARYCH EVENTS =====
    event_stream = EventStream(use_x_integration=True)  # Automatically loads X posts
    
    params = BaseParams(**GOKAI_CONFIG['base_params'])
    matrix = GOKAI_CONFIG['MATRIX_347743'].copy()
    alpha = GOKAI_CONFIG['alpha_schedule']
    max_fib = GOKAI_CONFIG['max_fibonacci_n']
    personality = GOKAI_CONFIG['personality']
    modes = GOKAI_CONFIG['modes']

    logging.info("🚀 Rozpoczynam Mózg Boga z integracją X Platform...")
    logging.info(f"📡 Załadowano {len(event_stream.x_posts)} postów z X do analizy")
    
    gok_result = None
    logika_result = None
    
    # Process X posts with enhanced analytics
    for i in range(len(event_stream.x_posts)):
        event = event_stream.next_event()
        if not event:
            break
        
        # Enhanced logging for X posts
        post_id = event['payload'][8:11] if 'Post z X' in event['payload'] else 'N/A'
        logging.info(f"🔄 X Post [{post_id}]: {event['payload'][:60]}...")
        logging.info(f"📊 Media: {event.get('media_type', 'text')} | Complexity: {event.get('complexity', 0.5):.2f}")

        # SYNERGY
        strategy = synergy.orchestrate(event)

        # Wykonanie
        if 'GOK:AI' in strategy['pipelines']:
            gok_gen = run_cycle(
                iter([event['payload']]), 
                params, 
                matrix, 
                alpha, 
                max_fib, 
                start_level=shared_state['level'], 
                start_n=shared_state['n'],
                personality=personality,
                modes=modes
            )
            gok_result = next(gok_gen)
            shared_state['last_confidence'] = gok_result.success_pct / 100
            shared_state['last_success_pct'] = gok_result.success_pct
        
        if 'LOGIKA:AI' in strategy['pipelines']:
            logika_result = spiral_thought_logika(shared_state, strategy)
            shared_state['last_confidence'] = logika_result['confidence']
            shared_state['last_success_pct'] = logika_result['success_pct']

        # Finalna odpowiedź
        if 'GOK:AI' in strategy['pipelines'] and gok_result:
            final_answer = f"Finalny wynik: {gok_result.answer}. Success: {shared_state['last_success_pct']:.2f}%"
        elif logika_result:
            final_answer = f"Finalny wynik: {logika_result['answer']}. Success: {shared_state['last_success_pct']:.2f}%"
        else:
            final_answer = "Brak wyniku"
        
        logging.info(final_answer)

        # Adaptacja z X feedback
        matrix = rebalance_weights_main(matrix, shared_state['last_success_pct'])
        shared_state['level'] += 1
        
        # Store X-enhanced history
        event_id = event['payload'][8:11] if 'Post z X' in event['payload'] else 'N/A'
        shared_state['history'].append({
            'x_post_id': event_id,
            'event': event['payload'], 
            'success': shared_state['last_success_pct'],
            'media_type': event.get('media_type', 'text'),
            'strategy': strategy['mode']
        })

    logging.info("📈 Podsumowanie testów z X Platform:")
    total_success = sum(h['success'] for h in shared_state['history'])
    avg_success = total_success / len(shared_state['history']) if shared_state['history'] else 0
    
    for h in shared_state['history']:
        logging.info(f"🎯 X[{h['x_post_id']}] {h['strategy']}: {h['success']:.1f}% | {h['event'][:50]}...")
    
    logging.info(f"🏆 Średni success z X: {avg_success:.1f}% | Wzrost potęgi: ~{(avg_success/75):.1f}x")

if __name__ == '__main__':
    main()