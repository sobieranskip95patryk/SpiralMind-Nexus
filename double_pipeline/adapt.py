import random
from typing import List
from .utils import BaseParams

def rebalance_weights(weights: List[int], success_history: List[float]) -> List[int]:
    """Rebalansuje wagi matrycy na podstawie historii sukcesów"""
    if not success_history:
        return weights
    
    avg_success = sum(success_history) / len(success_history)
    new_weights = list(weights)
    
    if avg_success > 90.0:
        # Zwiększ wszystkie wagi przy wysokim sukcesie
        new_weights = [min(9, w + 1) for w in new_weights]
    elif avg_success < 70.0:
        # Losowo zmień wagi przy niskim sukcesie
        rand_idx = random.randint(0, len(new_weights) - 1)
        new_weights[rand_idx] = max(1, new_weights[rand_idx] + random.choice([-2, 2]))
    
    return new_weights

def tune_base_params(params: BaseParams, performance_score: float) -> BaseParams:
    """Dostrajanie parametrów bazowych na podstawie wydajności"""
    if performance_score < 0.7:
        # Zwiększ parametry przy niskiej wydajności
        return BaseParams(
            W=min(9, params.W + 1),
            M=params.M,
            D=params.D,
            C=min(9, params.C + 1),
            A=params.A,
            E=params.E,
            T=params.T
        )
    return params
