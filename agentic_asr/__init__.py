"""Agentic ASR - Automatic Speech Recognition with LLM-powered agents."""

__version__ = "0.1.0"
__author__ = "Sergey Adamyan"

from .core.agent import SimpleAgent
from .core.history import HistoryManager
from .asr.transcriber import WhisperTranscriber
from .llm.providers import LLMProvider

__all__ = ["SimpleAgent", "HistoryManager", "WhisperTranscriber", "LLMProvider"]
