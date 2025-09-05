"""Logging configuration for Agentic ASR."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not format_string:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    
    handlers = [console_handler]
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
        )
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    logger = logging.getLogger("agentic_asr")
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def get_file_logger() -> logging.Logger:
    """Get file logger for detailed logging."""
    return logging.getLogger("agentic_asr.file")


def get_tool_logger() -> logging.Logger:
    """Get tool logger for tool execution logging."""
    return logging.getLogger("agentic_asr.tools")


log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", "./logs/agentic_asr.log")

logger = setup_logging(level=log_level, log_file=log_file)
