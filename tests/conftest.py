"""Test configuration and fixtures for SpiralMind-Nexus."""

import pytest
import tempfile
import os

from spiral.config.loader import Cfg, QuantumCfg, GOKAICfg, PipelineCfg
from spiral.core.quantum_core import QuantumCore
from spiral.core.gokai_core import GOKAICalculator
from spiral.pipeline.synergy_orchestrator import SynergyOrchestrator
from spiral.memory.persistence import MemoryPersistence


@pytest.fixture
def test_config():
    """Test configuration."""
    return Cfg(
        quantum=QuantumCfg(
            weights={
                'fibonacci': 0.3,
                'entropy': 0.25,
                'complexity': 0.25,
                's9': 0.2
            },
            cache_size=100
        ),
        gokai=GOKAICfg(
            weights={
                'quantum': 0.4,
                'semantic': 0.3,
                'contextual': 0.2,
                'temporal': 0.1
            },
            history_size=50
        ),
        pipeline=PipelineCfg(
            mode="quantum",
            batch_size=10,
            parallel_workers=2
        )
    )


@pytest.fixture
def quantum_core(test_config):
    """Quantum core instance."""
    return QuantumCore(test_config.quantum.__dict__)


@pytest.fixture
def gokai_calculator(test_config):
    """GOKAI calculator instance."""
    return GOKAICalculator(test_config.gokai.__dict__)


@pytest.fixture
def synergy_orchestrator(test_config):
    """Synergy orchestrator instance."""
    return SynergyOrchestrator(test_config)


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def memory_persistence(temp_db):
    """Memory persistence instance with temporary database."""
    return MemoryPersistence(temp_db)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello, world!",
        "This is a test of the quantum processing system.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once.",
        "Fibonacci sequences and Shannon entropy are fundamental concepts in mathematics and information theory.",
        "Complex text analysis involves multiple layers of linguistic and statistical processing to extract meaningful patterns and insights.",
        "",  # Empty text
        "A",  # Single character
        "Short",  # Short text
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10  # Long text
    ]


@pytest.fixture
def sample_contexts():
    """Sample contexts for testing."""
    return [
        {},
        {'importance': 0.8, 'urgency': 0.5},
        {'quality': 0.9, 'category': 'technical'},
        {'timestamp': '2023-01-01T00:00:00', 'source': 'test'},
        {'metadata': {'author': 'test', 'version': '1.0'}}
    ]


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.ERROR)  # Reduce noise in tests


@pytest.fixture
def temp_config_file():
    """Temporary configuration file."""
    import yaml
    
    config_data = {
        'quantum': {
            'weights': {
                'fibonacci': 0.3,
                'entropy': 0.25,
                'complexity': 0.25,
                's9': 0.2
            }
        },
        'gokai': {
            'weights': {
                'quantum': 0.4,
                'semantic': 0.3,
                'contextual': 0.2,
                'temporal': 0.1
            }
        },
        'pipeline': {
            'mode': 'quantum',
            'batch_size': 10
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)
