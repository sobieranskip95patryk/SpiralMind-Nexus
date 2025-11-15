"""Pipeline module for SpiralMind-Nexus.

This module provides pipeline processing and orchestration.
"""

from .synergy_orchestrator import SynergyOrchestrator
from .double_pipeline import execute, create_event

__all__ = ['SynergyOrchestrator', 'execute', 'create_event']
