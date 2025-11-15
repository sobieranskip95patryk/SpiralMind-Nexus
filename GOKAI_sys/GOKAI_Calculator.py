from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import numpy as np

class Archetype(Enum):
    HEAVEN = "Heaven"
    HELL = "Hell"

class DevelopmentPhase(Enum):
    DESTRUCTION = "Destrukcja"
    POINT_0 = "Punkt 0"
    DEVELOPMENT = "Rozwój"

class IntrinsicValue:
    def _init_(self): self.value = 7
    def get_value(self): return self.value

class SkillValue:
    def _init_(self): self.value = 6
    def get_value(self): return self.value

class DecisionValue:
    def _init_(self): self.value = 4
    def get_value(self): return self.value

class ContextValue:
    def _init_(self): self.value = 5
    def get_value(self): return self.value

class PersonalityValue:
    def _init_(self): self.value = 8
    def get_value(self): return self.value

class EnergyValue:
    def _init_(self): self.value = 6
    def get_value(self): return self.value

class IdentityValue:
    def _init_(self): self.value = 3
    def get_value(self): return self.value

@dataclass
class AIPsycheGOKAI:
    w: IntrinsicValue = IntrinsicValue()
    m: SkillValue = SkillValue()
    d: DecisionValue = DecisionValue()
    c: ContextValue = ContextValue()
    a: PersonalityValue = PersonalityValue()
    e: EnergyValue = EnergyValue()
    t: IdentityValue = IdentityValue()
    _iteration_count: int = 0

    def _parse_identity_matrix(self) -> List[int]:
        return [3, 6, 9, 9, 6, 3]

    def _transform_digit(self, digit: int, phase: DevelopmentPhase, prev_digit: int = 0, next_digit: int = 0) -> int:
        base_change = {
            DevelopmentPhase.DESTRUCTION: -1,
            DevelopmentPhase.POINT_0: 0,
            DevelopmentPhase.DEVELOPMENT: 1
        }[phase]
        adjusted_change = base_change + (prev_digit + next_digit - 10) / 10
        new_digit = max(1, min(9, digit + round(adjusted_change)))
        return new_digit

    def _evolve_identity_matrix(self, current_phase: DevelopmentPhase, iterations: int = 3) -> List[int]:
        matrix = self._parse_identity_matrix() if self._iteration_count == 0 else self._evolve_identity_matrix(current_phase, iterations - 1)
        
        for _ in range(iterations):
            new_matrix = matrix.copy()
            for i in range(len(matrix)):
                prev = matrix[(i - 1) % 6]
                curr = matrix[i]
                next_val = matrix[(i + 1) % 6]
                new_matrix[i] = self._transform_digit(curr, current_phase, prev, next_val)
            diff = 36 - sum(new_matrix)
            if diff != 0:
                idx = np.argmax(new_matrix)
                new_matrix[idx] += diff
            matrix = [max(1, min(9, x)) for x in new_matrix]
            self._iteration_count += 1
        
        return matrix

    def _evolve_identity(self, current_phase: DevelopmentPhase) -> Dict[str, float]:
        matrix = self._evolve_identity_matrix(current_phase)
        phase_index = {
            DevelopmentPhase.DESTRUCTION: 0,
            DevelopmentPhase.POINT_0: 2,
            DevelopmentPhase.DEVELOPMENT: 4
        }
        current_index = phase_index.get(current_phase, 0)
        
        identity_weights = {
            "W": matrix[(current_index + 0) % 6] / 9,
            "M": matrix[(current_index + 1) % 6] / 9,
            "D": matrix[(current_index + 2) % 6] / 9,
            "C": matrix[(current_index + 3) % 6] / 9,
            "A": matrix[(current_index + 4) % 6] / 9,
            "E": matrix[(current_index + 5) % 6] / 9,
            "T": matrix[current_index % 6] / 9
        }
        return identity_weights

    def assess_development_phase(self) -> DevelopmentPhase:
        energy_health = self.e.get_value() / 6
        decision_quality = self.d.get_value() / 4
        phase_score = (energy_health + decision_quality) / 2
        
        if phase_score < 0.3:
            return DevelopmentPhase.DESTRUCTION
        elif 0.3 <= phase_score < 0.7:
            return DevelopmentPhase.POINT_0
        else:
            return DevelopmentPhase.DEVELOPMENT

    def calculate_capital(self) -> float:
        return self.m.get_value() * (self.d.get_value() / 4)

    def predict_limit_boundary(self, historical_data: List[float]) -> float:
        if not historical_data:
            return 1.0
        trend = np.mean(historical_data[-3:])
        max_capacity = sum(self.w.get_value(), self.m.get_value(), self.a.get_value()) / 3
        return min(1.0, trend / max_capacity)

    def evaluate_decision_quality(self, past_decisions: List[Dict]) -> float:
        success_rate = sum(1 for d in past_decisions if d.get('success', False)) / len(past_decisions) if past_decisions else 0.5
        consistency = sum(1 for d in past_decisions if d.get('consistent', True)) / len(past_decisions) if past_decisions else 0.5
        return (success_rate + consistency) / 2

    def detect_disintegration_points(self, past_decisions: List[Dict]) -> List[Dict]:
        return [d for d in past_decisions if not d.get('consistent', True) or not d.get('success', False)]

    def calculate_success_probability(self, scenario: Dict) -> float:
        current_phase = self.assess_development_phase()
        identity_weights = self._evolve_identity(current_phase)

        w_weight = self.w.get_value() / 30 * identity_weights["W"]
        m_weight = self.m.get_value() / 30 * identity_weights["M"]
        a_weight = self.a.get_value() / 30 * identity_weights["A"]
        e_weight = self.e.get_value() / 30 * identity_weights["E"]

        energy_impact = min(1.0, self.e.get_value() / (self.m.get_value() * (self.c.get_value() ** 2)))
        synergy_factor = min(1.0, self.m.get_value() / 6)
        alignment = min(1.0, self.a.get_value() / 8)
        intent_match = min(1.0, self.w.get_value() / 7)

        base_probability = (energy_impact * e_weight + 
                          synergy_factor * m_weight + 
                          alignment * a_weight + 
                          intent_match * w_weight)
        phase_modifier = {
            DevelopmentPhase.DESTRUCTION: 0.5,
            DevelopmentPhase.POINT_0: 0.8,
            DevelopmentPhase.DEVELOPMENT: 1.2
        }[current_phase]

        return min(1.0, base_probability * phase_modifier)

    def generate_recommendations(self, scenarios: List[Dict]) -> List[Dict]:
        recommendations = []
        for scenario in scenarios:
            prob_success = self.calculate_success_probability(scenario)
            recommendations.append({
                "scenario": scenario,
                "probability": prob_success,
                "phase_context": self.assess_development_phase().value,
                "capital_utilization": self.calculate_capital(),
                "limit_boundary": self.predict_limit_boundary([s.get('outcome', 1.0) for s in scenarios]),
                "identity_matrix": self._evolve_identity_matrix(self.assess_development_phase()),
                "identity_weights": self._evolve_identity(self.assess_development_phase())
            })
        return sorted(recommendations, key=lambda x: x["probability"], reverse=True)

if __name__ == "__main__":
    psyche = AIPsycheGOKAI()

    print("AI_Psyche_GOK:AI - Psychologia Prawdopodobieństw Sukcesu")
    print(f"Bieżąca faza rozwoju: {psyche.assess_development_phase().value}")
    print(f"Zgromadzony kapitał: {psyche.calculate_capital()}")
    print(f"Przewidywana granica możliwości: {psyche.predict_limit_boundary([]):.2f}")

    past_decisions = [
        {"success": True, "consistent": True},
        {"success": False, "consistent": False},
        {"success": True, "consistent": True}
    ]
    print(f"Jakość decyzji z przeszłości: {psyche.evaluate_decision_quality(past_decisions):.2f}")
    print(f"Punkty dezintegracji: {len(psyche.detect_disintegration_points(past_decisions))}")

    scenarios = [
        {"goal": "Innowacja", "resources": 5, "outcome": 0.9},
        {"goal": "Optymalizacja", "resources": 3, "outcome": 0.7}
    ]
    recommendations = psyche.generate_recommendations(scenarios)
    print("\nRekomendacje:")
    for rec in recommendations:
        print(f"Scenariusz: {rec['scenario']['goal']}, Prawdopodobieństwo sukcesu: {rec['probability']:.2f}, "
              f"Faza: {rec['phase_context']}, Kapitał: {rec['capital_utilization']:.2f}, "
              f"Granica możliwości: {rec['limit_boundary']:.2f}, "
              f"Matryca tożsamości: {rec['identity_matrix']}, Wagi tożsamości: {rec['identity_weights']}")