from setuptools import setup, find_packages

setup(
    name="agentic-asr",
    version="0.1.0",
    description="Automatic Speech Recognition with LLM-powered agents",
    author="Sergey Adamyan",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pydub>=0.25.1",
        "ffmpeg-python>=0.2.0",
        "python-dotenv>=1.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "agentic-asr=agentic_asr.cli:main",
        ],
    },
)
