"""Core module for SpiralMind-Nexus.

This module contains the core quantum and GOKAI components.
"""

from .quantum_core import QuantumCore
from .gokai_core import GOKAICalculator, Score

__all__ = ['QuantumCore', 'GOKAICalculator', 'Score']
__version__ = '0.2.0'
