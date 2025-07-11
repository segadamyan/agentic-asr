# Agentic ASR

Automatic Speech Recognition with LLM-powered agents for intelligent audio processing.

## Features

- **Speech Recognition**: High-quality transcription using OpenAI Whisper
- **LLM Integration**: Support for OpenAI GPT and Anthropic Claude models
- **Intelligent Processing**: AI-powered analysis, summarization, and keyword extraction
- **Conversation History**: Persistent storage of interactions and context
- **Tool Integration**: Extensible tool system for custom processing
- **CLI Interface**: Easy-to-use command-line tools
- **Session Management**: Continue conversations across sessions

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

### Install FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official site](https://ffmpeg.org/download.html)

### Install Agentic ASR

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-asr.git
cd agentic-asr

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/agentic_asr.db

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=auto

# Default LLM Configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4
```

## Quick Start

### 1. Transcribe Audio

```bash
# Basic transcription
agentic-asr transcribe audio.wav

# Specify model and language
agentic-asr transcribe audio.wav --model small --language en

# Save to file
agentic-asr transcribe audio.wav --output transcription.txt
```

### 2. Interactive Chat

```bash
# Start interactive session
agentic-asr chat

# Continue existing session
agentic-asr chat --session your-session-id

# Use different LLM
agentic-asr chat --provider anthropic --model claude-3-sonnet-20240229
```

### 3. Process Audio with AI

```bash
# Transcribe and ask questions
agentic-asr process audio.wav "What are the main topics discussed?"

# Analyze and summarize
agentic-asr process audio.wav "Summarize this audio in bullet points"
```

### 4. Manage Sessions

```bash
# List recent sessions
agentic-asr list-sessions

# Show version
agentic-asr version
```

## Python API Usage

### Basic Transcription

```python
import asyncio
from agentic_asr.asr.transcriber import create_transcriber

async def transcribe_audio():
    transcriber = create_transcriber(model_name="base")
    result = await transcriber.transcribe_file("audio.wav")
    
    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence}")

asyncio.run(transcribe_audio())
```

### Agent Interaction

```python
import asyncio
import os
from agentic_asr.core.agent import create_asr_agent
from agentic_asr.core.models import LLMProviderConfig

async def chat_with_agent():
    # Configure LLM
    llm_config = LLMProviderConfig(
        provider_name="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create agent
    agent = await create_asr_agent(llm_config=llm_config)
    
    # Chat
    response = await agent.answer_to("What can you help me with?")
    print(response.content)
    
    await agent.close()

asyncio.run(chat_with_agent())
```

## Architecture

The system is built with a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Python API     │    │   Web Interface │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
              ┌─────────────────────────────────────┐
              │           SimpleAgent               │
              ├─────────────────────────────────────┤
              │ - Conversation Management           │
              │ - Tool Execution                    │
              │ - History Persistence               │
              │ - Token Management                  │
              └─────────────────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼───┐              ┌────────▼────────┐          ┌───────▼───────┐
│  ASR  │              │  LLM Providers  │          │     Tools     │
│Module │              │                 │          │               │
├───────┤              ├─────────────────┤          ├───────────────┤
│Whisper│              │ OpenAI GPT      │          │ Transcription │
│Models │              │ Anthropic Claude│          │ Summarization │
└───────┘              └─────────────────┘          │ Keywords      │
                                                    │ Custom Tools  │
                                                    └───────────────┘
```

## Available Tools

The agent comes with built-in tools:

- **transcribe_audio**: Convert audio files to text
- **summarize_text**: Generate summaries of transcribed content
- **extract_keywords**: Extract key terms and phrases
- **Custom tools**: Extend with your own processing functions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
