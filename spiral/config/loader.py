"""Configuration loader for SpiralMind-Nexus.

Provides YAML configuration loading with dataclass validation.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumCfg:
    """Quantum processing configuration."""
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        'fibonacci': 0.3,
        'entropy': 0.25,
        'complexity': 0.25,
        's9': 0.2
    })
    cache_size: int = 1000
    enable_parallel: bool = True
    batch_size: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure weights sum to approximately 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Quantum weights sum to {weight_sum:.3f}, expected ~1.0")
        
        # Validate weight values
        for key, weight in self.weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Weight '{key}' must be between 0.0 and 1.0, got {weight}")


@dataclass
class GOKAICfg:
    """GOKAI calculator configuration."""
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        'quantum': 0.4,
        'semantic': 0.3,
        'contextual': 0.2,
        'temporal': 0.1
    })
    confidence: Dict[str, float] = field(default_factory=lambda: {
        'base_confidence': 0.7,
        'length_factor': 0.1,
        'complexity_factor': 0.15,
        'consistency_factor': 0.05
    })
    history_size: int = 100
    min_confidence: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"GOKAI weights sum to {weight_sum:.3f}, expected ~1.0")
            
        # Validate confidence parameters
        if not 0.0 <= self.confidence['base_confidence'] <= 1.0:
            raise ValueError("base_confidence must be between 0.0 and 1.0")
            
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass
class PipelineCfg:
    """Pipeline configuration."""
    
    mode: str = "quantum"
    batch_size: int = 50
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    parallel_workers: int = 4
    timeout: float = 30.0  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_modes = ['quantum', 'gokai', 'hybrid', 'debug']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}', must be one of {valid_modes}")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        if self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
            
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class LoggingCfg:
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level '{self.level}', must be one of {valid_levels}")
        
        self.level = self.level.upper()


@dataclass
class APICfg:
    """API configuration."""
    
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100  # requests per minute
    max_request_size: int = 1024 * 1024  # 1MB
    enable_docs: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
            
        if self.rate_limit <= 0:
            raise ValueError("rate_limit must be positive")


@dataclass
class DatabaseCfg:
    """Database configuration."""
    
    url: str = "sqlite:///spiral_memory.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    echo: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
            
        if self.pool_timeout <= 0:
            raise ValueError("pool_timeout must be positive")


@dataclass
class Cfg:
    """Main configuration class."""
    
    quantum: QuantumCfg = field(default_factory=QuantumCfg)
    gokai: GOKAICfg = field(default_factory=GOKAICfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    api: APICfg = field(default_factory=APICfg)
    database: DatabaseCfg = field(default_factory=DatabaseCfg)
    
    # Global settings
    version: str = "0.2.0"
    environment: str = "development"
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_environments = ['development', 'testing', 'production']
        if self.environment not in valid_environments:
            raise ValueError(f"Invalid environment '{self.environment}', must be one of {valid_environments}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cfg':
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Cfg instance
        """
        # Extract nested configurations
        quantum_data = data.get('quantum', {})
        gokai_data = data.get('gokai', {})
        pipeline_data = data.get('pipeline', {})
        logging_data = data.get('logging', {})
        api_data = data.get('api', {})
        database_data = data.get('database', {})
        
        # Create nested config objects
        quantum_cfg = QuantumCfg(**quantum_data)
        gokai_cfg = GOKAICfg(**gokai_data)
        pipeline_cfg = PipelineCfg(**pipeline_data)
        logging_cfg = LoggingCfg(**logging_data)
        api_cfg = APICfg(**api_data)
        database_cfg = DatabaseCfg(**database_data)
        
        # Extract global settings
        global_data = {k: v for k, v in data.items() 
                      if k not in ['quantum', 'gokai', 'pipeline', 'logging', 'api', 'database']}
        
        return cls(
            quantum=quantum_cfg,
            gokai=gokai_cfg,
            pipeline=pipeline_cfg,
            logging=logging_cfg,
            api=api_cfg,
            database=database_cfg,
            **global_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'quantum': self.quantum.__dict__,
            'gokai': self.gokai.__dict__,
            'pipeline': self.pipeline.__dict__,
            'logging': self.logging.__dict__,
            'api': self.api.__dict__,
            'database': self.database.__dict__,
            'version': self.version,
            'environment': self.environment,
            'debug': self.debug
        }


def load_config(config_path: Optional[str] = None) -> Cfg:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, looks for default locations.
        
    Returns:
        Loaded configuration
        
    Raises:
        FileNotFoundError: If configuration file is not found
        yaml.YAMLError: If configuration file is invalid YAML
        ValueError: If configuration is invalid
    """
    # Default configuration file locations
    default_paths = [
        config_path,
        os.environ.get('SPIRAL_CONFIG'),
        'config/config.yaml',
        'config.yaml',
        'spiral_config.yaml',
        Path.home() / '.spiral' / 'config.yaml'
    ]
    
    config_file = None
    for path in default_paths:
        if path and os.path.exists(path):
            config_file = path
            break
    
    if config_file:
        logger.info(f"Loading configuration from {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            if not isinstance(data, dict):
                raise ValueError("Configuration file must contain a YAML dictionary")
                
            return Cfg.from_dict(data)
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    else:
        logger.info("No configuration file found, using defaults")
        return Cfg()


def save_config(config: Cfg, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def merge_configs(*configs: Cfg) -> Cfg:
    """Merge multiple configurations.
    
    Args:
        configs: Configuration objects to merge
        
    Returns:
        Merged configuration (later configs override earlier ones)
    """
    if not configs:
        return Cfg()
        
    # Start with first config
    result_dict = configs[0].to_dict()
    
    # Merge remaining configs
    for config in configs[1:]:
        config_dict = config.to_dict()
        _deep_merge(result_dict, config_dict)
    
    return Cfg.from_dict(result_dict)


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    """Deep merge dict2 into dict1.
    
    Args:
        dict1: Target dictionary (modified in place)
        dict2: Source dictionary
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            _deep_merge(dict1[key], value)
        else:
            dict1[key] = value
