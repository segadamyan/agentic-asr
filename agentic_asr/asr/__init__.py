"""ASR module initialization."""

from .transcriber import WhisperTranscriber, create_transcriber

__all__ = ["WhisperTranscriber", "create_transcriber"]
