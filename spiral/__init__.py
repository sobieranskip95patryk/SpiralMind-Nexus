"""
SpiralMind Nexus - AI Double Pipeline System
Version 0.2.0 - Production Ready
"""

__version__ = "0.2.0"
__author__ = "Patryk Sobierański META-GENIUSZ®️🇮🇩"

from .core import quantum_core, gokai_core
from .config import loader
from .pipeline import synergy_orchestrator, double_pipeline

__all__ = [
    "quantum_core",
    "gokai_core", 
    "loader",
    "synergy_orchestrator",
    "double_pipeline"
]