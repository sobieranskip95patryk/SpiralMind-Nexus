from typing import Dict

# Modyfikatory dla typu INTP
INTP_MODS = {
    "alpha_boost": 0.15,
    "logic_bias": 0.8,
    "abstract_thinking": 0.9,
    "pattern_recognition": 0.85,
    "deep_focus": True
}

# Modyfikatory dla trybu Einstein
EINSTEIN_MODS = {
    "theoretical_bias": 0.9,
    "thought_experiment": 0.8,
    "relativity_thinking": 0.7,
    "paradox_resolution": 0.85,
    "cosmic_perspective": True
}

# Modyfikatory dla trybu Gates
GATES_MODS = {
    "system_optimization": 0.9,
    "scalability_focus": 0.8,
    "efficiency_drive": 0.85,
    "strategic_thinking": 0.9,
    "innovation_bias": 0.7
}

def merge_mods(mods1: Dict[str, float], mods2: Dict[str, float]) -> Dict[str, float]:
    """{łączy dwa słowniki modyfikatorów"""
    merged = dict(mods1)
    for key, value in mods2.items():
        if key in merged and isinstance(merged[key], (int, float)) and isinstance(value, (int, float)):
            merged[key] = (merged[key] + value) / 2  # średnia
        else:
            merged[key] = value
    return merged

def get_personality_traits(personality_type: str) -> Dict[str, float]:
    """Zwraca cechy osobowości dla danego typu"""
    traits_map = {
        "INTP": {
            "logic": 0.9,
            "intuition": 0.8,
            "thinking": 0.85,
            "perceiving": 0.7,
            "introversion": 0.6
        },
        "INTJ": {
            "logic": 0.85,
            "intuition": 0.9,
            "thinking": 0.8,
            "judging": 0.8,
            "introversion": 0.7
        }
    }
    return traits_map.get(personality_type, {})
