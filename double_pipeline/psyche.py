from dataclasses import dataclass

@dataclass
class PsycheSignal:
    bias: float
    valence: float
    intent: str

def normalize_payload(payload: str) -> str:
    return payload.strip()

def extract_psych_metrics_v2(doc: str, context: dict, risk_bias: float = 0.0) -> PsycheSignal:
    length = len(doc)
    valence = 0.2 if "sukces" in doc.lower() else 0.0
    bias = min(10.0, length / 100.0) * (1.0 + risk_bias)
    intent = "analysis" if length > 30 else "ping"
    return PsycheSignal(bias=bias, valence=valence, intent=intent)

def scale_signal(sig: PsycheSignal, weight: int) -> PsycheSignal:
    return PsycheSignal(bias=sig.bias * weight, valence=sig.valence, intent=sig.intent)

def personality_to_modulators(personality: dict) -> dict:
    mind_N = personality.get("mind", {}).get("intuitive", 50)
    tactics_P = personality.get("tactics", {}).get("prospecting", 50)
    nature_T = personality.get("nature", {}).get("thinking", 50)
    nature_F = personality.get("nature", {}).get("feeling", 50)
    identity_Turb = personality.get("identity", {}).get("turbulent", 50)
    return {
        "alpha_boost": 0.002 * (mind_N + tactics_P),
        "logic_bias": nature_T / 100.0,
        "chaos_bias": nature_F / 100.0,
        "risk_bias": (identity_Turb / 100.0) * 0.5
    }

def apply_personality(sig: PsycheSignal, mods: dict) -> PsycheSignal:
    new_bias = sig.bias * (1.0 + mods.get("risk_bias", 0.0)) * (1.0 + mods.get("logic_bias", 0.0) * 0.1)
    new_val = sig.valence + (mods.get("chaos_bias", 0.0) - mods.get("logic_bias", 0.0)) * 0.1
    return PsycheSignal(bias=new_bias, valence=new_val, intent=sig.intent)
