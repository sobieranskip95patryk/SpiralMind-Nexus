"""Logging configuration for SpiralMind-Nexus.

Provides centralized logging setup and management.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    logger_name: str = "spiral"
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Optional custom log format
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        enable_console: Whether to enable console logging
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level = level.upper()
    if level not in valid_levels:
        level = 'INFO'
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger for the application
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
            
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {level}, Console: {enable_console}, File: {bool(log_file)}")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'spiral')
    
    # Ensure it's a child of the main spiral logger
    if not name.startswith('spiral'):
        name = f"spiral.{name}"
    
    return logging.getLogger(name)


def configure_third_party_loggers(level: str = "WARNING") -> None:
    """Configure third-party library loggers to reduce noise.
    
    Args:
        level: Log level for third-party loggers
    """
    third_party_loggers = [
        'urllib3',
        'requests',
        'asyncio',
        'concurrent.futures',
        'sqlite3'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


def setup_development_logging() -> logging.Logger:
    """Setup logging for development environment.
    
    Returns:
        Configured logger for development
    """
    return setup_logging(
        level="DEBUG",
        log_file="logs/spiral_dev.log",
        enable_console=True
    )


def setup_production_logging() -> logging.Logger:
    """Setup logging for production environment.
    
    Returns:
        Configured logger for production
    """
    return setup_logging(
        level="INFO",
        log_file="logs/spiral_prod.log",
        enable_console=False,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )


def setup_testing_logging() -> logging.Logger:
    """Setup logging for testing environment.
    
    Returns:
        Configured logger for testing
    """
    return setup_logging(
        level="ERROR",
        enable_console=False
    )


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging.
    
    Args:
        logger: Logger to use for output
    """
    import platform
    
    try:
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"Architecture: {platform.architecture()[0]}")
        logger.info(f"CPU Count: {os.cpu_count()}")
        logger.info(f"Working Directory: {os.getcwd()}")
        
    except Exception as e:
        logger.warning(f"Could not log system info: {e}")


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, 
                 logger: logging.Logger,
                 level: str = None,
                 extra_context: Dict[str, Any] = None):
        """Initialize logging context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
            extra_context: Extra context to add to log messages
        """
        self.logger = logger
        self.original_level = logger.level
        self.new_level = getattr(logging, level.upper()) if level else None
        self.extra_context = extra_context or {}
        self.original_filter = None
    
    def __enter__(self):
        if self.new_level:
            self.logger.setLevel(self.new_level)
        
        if self.extra_context:
            # Add context filter
            class ContextFilter(logging.Filter):
                def __init__(self, context):
                    super().__init__()
                    self.context = context
                
                def filter(self, record):
                    for key, value in self.context.items():
                        setattr(record, key, value)
                    return True
            
            self.context_filter = ContextFilter(self.extra_context)
            self.logger.addFilter(self.context_filter)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.new_level:
            self.logger.setLevel(self.original_level)
        
        if hasattr(self, 'context_filter'):
            self.logger.removeFilter(self.context_filter)


def with_logging_context(logger: logging.Logger, **context):
    """Decorator to add logging context to functions.
    
    Args:
        logger: Logger to use
        **context: Context to add to log messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingContext(logger, extra_context=context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class TimingLogger:
    """Logger for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        """Initialize timing logger.
        
        Args:
            logger: Logger to use
            operation: Description of operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type:
                self.logger.error(f"{self.operation} failed after {duration:.3f}s: {exc_val}")
            else:
                self.logger.info(f"{self.operation} completed in {duration:.3f}s")


def time_operation(logger: logging.Logger, operation: str):
    """Create a timing context manager.
    
    Args:
        logger: Logger to use
        operation: Description of operation
        
    Returns:
        TimingLogger context manager
    """
    return TimingLogger(logger, operation)


# Module-level logger for this package
_logger = None

def get_module_logger() -> logging.Logger:
    """Get the module-level logger.
    
    Returns:
        Module logger
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


# Auto-configure logging based on environment
def _auto_configure():
    """Auto-configure logging based on environment variables."""
    env = os.environ.get('SPIRAL_ENV', 'development').lower()
    
    if env == 'production':
        setup_production_logging()
    elif env == 'testing':
        setup_testing_logging()
    else:
        setup_development_logging()
    
    # Configure third-party loggers
    configure_third_party_loggers()


# Auto-configure on module import
if not os.environ.get('SPIRAL_NO_AUTO_LOGGING'):
    _auto_configure()
