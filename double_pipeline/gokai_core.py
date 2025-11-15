from typing import Iterator, Dict, Any
from dataclasses import dataclass
import random
import math
import os
from .utils import BaseParams, apply_formula_S, mix, map_to_range, clamp, shannon_entropy, fib
from .psyche import PsycheSignal, extract_psych_metrics_v2, personality_to_modulators, apply_personality, scale_signal
from .memory import ShortTermMemory, LongTermMemory, EpisodicMemory
from .adapt import rebalance_weights
from .multimodal import image_to_signals
from .multimodal_x import analyze_x_media
from .traits import INTP_MODS, EINSTEIN_MODS, GATES_MODS, merge_mods

@dataclass
class StepOutput:
    level: int
    stage: int
    n: int
    S9: int
    S_pi: float
    Fn: int
    wynik: float
    answer: str
    success_pct: float

def run_cycle(events: Iterator[str], base_params: BaseParams, matrix_weights: list, 
             alpha_schedule: list, max_fib_n: int, start_n: int = 1, 
             start_level: int = 0, personality: dict = None, modes: dict = None):
    level, n = start_level, start_n
    stm = ShortTermMemory(maxlen=128)
    ltm = LongTermMemory()
    epi = EpisodicMemory()
    success_hist = []
    extra_mods = {}
    
    if modes:
        if modes.get("enable_intp_mode"):
            extra_mods = merge_mods(extra_mods, INTP_MODS)
        if modes.get("enable_einstein_mode"):
            extra_mods = merge_mods(extra_mods, EINSTEIN_MODS)
        if modes.get("enable_gates_mode"):
            extra_mods = merge_mods(extra_mods, GATES_MODS)
    
    mods = personality_to_modulators(personality) if personality else {}
    mods = merge_mods(mods, extra_mods) if extra_mods else mods

    while True:
        for stage in range(7):
            weight = matrix_weights[stage % len(matrix_weights)]
            payload = next(events, '')
            if not payload:
                break
            
            # ===== INTEGRACJA X MEDIA =====
            current_event = payload if isinstance(payload, dict) else {'payload': payload}
            doc = current_event.get('payload', '').strip()
            entropy = shannon_entropy(doc)
            
            # Sprawdź czy to X post z mediami
            if 'Post z X' in doc:
                x_media_signals = analyze_x_media(current_event)
                entropy += x_media_signals.get("entropy", 0.0)
                n += int(x_media_signals.get("fib_hint", 0))
                # Dodaj X-specific boost do modulatorów
                if x_media_signals.get("x_relevance", 0) > 0.8:
                    mods["x_boost"] = x_media_signals["x_relevance"] * 0.1
            
            elif looks_like_image_path(doc):
                sig = image_to_signals(doc)
                entropy += sig.get("entropy", 0.0)
                n += int(sig.get("fib_hint", 0))
            
            stm.put("input", {"doc": doc, "entropy": entropy})

            psy_raw = extract_psych_metrics_v2(
                doc, 
                {"level": level, "n": n}, 
                risk_bias=max(0.0, 1.0 - stm.get_last("confidence", {"value": 0.75})["value"])
            )
            psyche = apply_personality(psy_raw, mods)
            psyche = scale_signal(psyche, weight)

            res = apply_formula_S(n, base_params)
            alpha = alpha_schedule[stage % len(alpha_schedule)] + min(0.3, entropy * 0.02) + mods.get("alpha_boost", 0.0)
            
            if extra_mods.get("deep_focus", False) and entropy > 2.5:
                alpha = max(0.0, alpha - 0.15)
            
            potency = 0.5  # Można rozszerzyć
            wynik_mixed = mix(res["WYNIK"], random.uniform(-1.0, 1.0) * potency * 3.0, alpha)

            max_target = 9 * math.pi + fib(max_fib_n)
            success = clamp(
                map_to_range(wynik_mixed, 0, max_target, 0, 100) + level * 1.5 + psyche.bias, 
                0, 100
            )
            
            confidence_score = max(0.0, min(1.0, 1.0 - abs(wynik_mixed - res["WYNIK"]) / (1.0 + res["WYNIK"])))
            
            answer = f"[L{level}/S{stage}/n{n}] WYNIK≈{wynik_mixed:.3f} | S9={res['S9']} Sπ={res['S_pi']:.3f} Fn={res['Fn']} | psyche.bias={psyche.bias:.2f}"

            yield StepOutput(
                level=level, stage=stage, n=n, S9=res["S9"], S_pi=res["S_pi"], 
                Fn=res["Fn"], wynik=wynik_mixed, answer=answer, success_pct=success
            )

            stm.put("confidence", {"value": confidence_score})
            epi.add(level=level, stage=stage, n=n, success=success, conf=confidence_score)
            ltm.upsert("last_answer", answer)
            success_hist.append(success)
            n += 1
        
        level += 1
        matrix_weights[:] = rebalance_weights(matrix_weights, success_hist)
        success_hist.clear()

def looks_like_image_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}