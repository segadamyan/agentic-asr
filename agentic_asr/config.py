"""Configuration management for agentic-asr."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Central configuration class for all app settings."""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Data subdirectories
    TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
    UPLOADS_DIR = DATA_DIR / "uploads"
    DOWNLOADS_DIR = DATA_DIR / "downloads"
    FAISS_INDICES_DIR = DATA_DIR / "faiss_indices"
    
    # Logs
    LOGS_DIR = PROJECT_ROOT / "logs"
    LOG_FILE = LOGS_DIR / "agentic_asr.log"
    
    # Database
    DATABASE_PATH = DATA_DIR / "agentic_asr.db"
    
    # Vector Store Settings
    VECTOR_STORE_MODEL = os.getenv("VECTOR_STORE_MODEL", "all-MiniLM-L6-v2")
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
    DEFAULT_LLM_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.3"))
    
    # Transcription Settings
    TRANSCRIPTION_MAX_SIZE_MB = int(os.getenv("TRANSCRIPTION_MAX_SIZE_MB", "500"))
    
    # RAG Settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.1"))
    MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
    
    # Chunking Settings (for LLM context limits)
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "30000"))
    CORRECTION_MAX_TOKENS = int(os.getenv("CORRECTION_MAX_TOKENS", "2000"))
    CORRECTION_CHUNK_TOKENS = int(os.getenv("CORRECTION_CHUNK_TOKENS", "1500"))
    SUMMARIZATION_MAX_TOKENS = int(os.getenv("SUMMARIZATION_MAX_TOKENS", "10000"))
    SUMMARIZATION_CHUNK_TOKENS = int(os.getenv("SUMMARIZATION_CHUNK_TOKENS", "8000"))
    TRANSLATION_MAX_TOKENS = int(os.getenv("TRANSLATION_MAX_TOKENS", "2000"))
    TRANSLATION_CHUNK_TOKENS = int(os.getenv("TRANSLATION_CHUNK_TOKENS", "1500"))
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Frontend Settings
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.TRANSCRIPTIONS_DIR,
            cls.UPLOADS_DIR,
            cls.DOWNLOADS_DIR,
            cls.FAISS_INDICES_DIR,
            cls.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_transcription_paths(cls) -> list[Path]:
        """Get list of possible transcription directory paths."""
        return [
            cls.TRANSCRIPTIONS_DIR,  # Primary path
            Path("../data/transcriptions"),  # From api directory
            Path("data/transcriptions"),      # From project root (relative)
        ]
    
    @classmethod
    def validate_llm_config(cls) -> bool:
        """Validate that required LLM API keys are present."""
        if cls.DEFAULT_LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            return False
        if cls.DEFAULT_LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            return False
        return True
    
    @classmethod
    def get_env_template(cls) -> str:
        """Get a template .env file content."""
        return """# LLM Provider Settings
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o
DEFAULT_LLM_TEMPERATURE=0.3

# Vector Store Settings
VECTOR_STORE_MODEL=all-MiniLM-L6-v2
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# RAG Settings
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.1
MAX_CONTEXT_CHUNKS=3

# Token Limits for Different Operations
DEFAULT_MAX_TOKENS=30000
CORRECTION_MAX_TOKENS=2000
CORRECTION_CHUNK_TOKENS=1500
SUMMARIZATION_MAX_TOKENS=10000
SUMMARIZATION_CHUNK_TOKENS=8000
TRANSLATION_MAX_TOKENS=2000
TRANSLATION_CHUNK_TOKENS=1500

# File Size Limits
TRANSCRIPTION_MAX_SIZE_MB=500

# API Settings
API_HOST=localhost
API_PORT=8000
API_DEBUG=false

# Frontend Settings
FRONTEND_PORT=3000
"""


# Initialize directories when config is imported
Config.ensure_directories()
