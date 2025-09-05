# Agentic ASR

Intelligent content processing system with LLM-powered agents for transcribed speech analysis.

## Features

- **Transcription Analysis**: Intelligent processing of transcribed text content
- **LLM Integration**: Support for OpenAI GPT and Anthropic Claude models
- **Intelligent Processing**: AI-powered analysis, summarization, and keyword extraction
- **Conversation History**: Persistent storage of interactions and context
- **Tool Integration**: Extensible tool system for custom processing
- **CLI Interface**: Easy-to-use command-line tools
- **Session Management**: Continue conversations across sessions
- **RAG Support**: Semantic search and retrieval-augmented generation
- **Vector Store**: Built-in FAISS-based vector storage for semantic search

## Installation

### Prerequisites

- Python 3.8 or higher

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

# Default LLM Configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4
```

## Quick Start

### 1. Analyze Transcriptions

```bash
# Analyze transcription files
agentic-asr analyze transcription.txt

# Summarize content
agentic-asr summarize transcription.txt

# Extract insights
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
agentic-asr extract-keywords transcription.txt
```

### 2. Interactive Sessions

```bash
# Start interactive session
agentic-asr chat

# Use with specific LLM provider
agentic-asr chat --provider openai --model gpt-4
```

### 3. Process with Context

```bash
# Analyze transcriptions with questions
agentic-asr process transcription.txt "What are the main topics discussed?"

# Analyze and summarize
agentic-asr process transcription.txt "Summarize this content in bullet points"
```

### 4. Manage Sessions

```bash
# List recent sessions
agentic-asr list-sessions

# Show version
agentic-asr version
```

## Python API Usage

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
│Vector │              │  LLM Providers  │          │     Tools     │
│Store  │              │                 │          │               │
├───────┤              ├─────────────────┤          ├───────────────┤
│FAISS  │              │ OpenAI GPT      │          │ Text Analysis │
│Search │              │ Anthropic Claude│          │ Summarization │
└───────┘              └─────────────────┘          │ Translation   │
                                                    │ RAG Search    │
                                                    └───────────────┘
```

## Available Tools

The agent comes with built-in tools:

- **analyze_transcription**: Analyze transcribed text content
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
