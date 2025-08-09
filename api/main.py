"""
FastAPI backend for Agentic ASR Web Interface.

This module provides REST API endpoints for:
- Transcription management
- Conversation history
- Agent interactions
- File management
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import from the main agentic_asr module
import sys
sys.path.append('..')

from agentic_asr.core.agent import create_asr_agent
from agentic_asr.core.history import HistoryManager
from agentic_asr.core.models import LLMProviderConfig

# Try to import optional transcription features
try:
    from agentic_asr.asr.transcriber import create_transcriber
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    print("Warning: Audio transcription features not available. Install audio dependencies for full functionality.")

from agentic_asr.tools.enhanced import (
    analyze_transcription_tool,
    correct_transcription_tool,
    summarize_transcription_file_tool,
    translate_transcription_file_tool
)

# Pydantic models for API requests/responses
class TranscriptionRequest(BaseModel):
    file_path: str
    language: Optional[str] = None
    model: str = "base"

class TranscriptionResponse(BaseModel):
    id: str
    text: str
    language: Optional[str]
    confidence: Optional[float]
    created_at: datetime
    metadata: Dict[str, Any]

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: List[Dict[str, Any]]

class AnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "summary"  # summary, keywords, sentiment

class CorrectionRequest(BaseModel):
    text: str
    context: str = ""
    correction_level: str = "medium"  # light, medium, heavy

class SummarizationRequest(BaseModel):
    filename: str
    summary_type: str = "comprehensive"
    extract_actions: bool = True
    extract_key_points: bool = True
    max_summary_length: int = 500

class TranslationRequest(BaseModel):
    filename: str
    target_language: str = "en"
    source_language: Optional[str] = None

# FastAPI app configuration
app = FastAPI(
    title="Agentic ASR API",
    description="REST API for Intelligent Speech Recognition with LLM processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",  # React dev server
        "https://segadamyan.github.io",  # GitHub Pages
        "https://*.railway.app",  # Railway deployment
        "https://*.vercel.app",   # Vercel deployment
        "https://*.herokuapp.com"  # Heroku deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for shared resources
history_manager: Optional[HistoryManager] = None
active_agents: Dict[str, Any] = {}

# Configuration
DATA_DIR = Path("../data")
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize shared resources on startup."""
    global history_manager
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/agentic_asr.db")
    history_manager = HistoryManager(database_url)
    await history_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global history_manager, active_agents
    
    # Close all active agents
    for agent in active_agents.values():
        try:
            await agent.close()
        except Exception as e:
            print(f"Error closing agent: {e}")
    
    # Close history manager
    if history_manager:
        await history_manager.close()

# Helper functions
def get_llm_config() -> LLMProviderConfig:
    """Get LLM configuration from environment."""
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported LLM provider: {provider}")
    
    return LLMProviderConfig(
        provider_name=provider,
        model=model,
        api_key=api_key,
        temperature=0.7
    )

async def get_or_create_agent(session_id: Optional[str] = None) -> Any:
    """Get existing agent or create new one."""
    global active_agents
    
    if session_id and session_id in active_agents:
        return active_agents[session_id]
    
    # Get available transcription files for system prompt
    available_files = []
    if TRANSCRIPTIONS_DIR.exists():
        available_files = [f.name for f in TRANSCRIPTIONS_DIR.glob("*.txt") if f.is_file()]
    
    # Create enhanced system prompt with available files
    files_context = ""
    if available_files:
        files_list = ", ".join(available_files[:15])  # Show first 15 files
        if len(available_files) > 15:
            files_list += f" (and {len(available_files) - 15} more)"
        files_context = f"\n\nAvailable transcription files in the system: {files_list}"
    
    system_prompt = f"""You are an intelligent ASR assistant that helps process and analyze transcribed speech. 
You can transcribe audio, summarize text, extract keywords, correct transcriptions, and translate content.
Use the available tools to help users with their transcription-related tasks.{files_context}"""
    
    # Create new agent
    llm_config = get_llm_config()
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/agentic_asr.db")
    agent = await create_asr_agent(
        system_prompt=system_prompt,
        llm_config=llm_config,
        database_url=database_url,
        session_id=session_id
    )
    
    active_agents[agent.session_id] = agent
    return agent

# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Agentic ASR API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

# Transcription endpoints
@app.get("/transcriptions", response_model=List[Dict[str, Any]])
async def list_transcriptions():
    """List all transcription files."""
    transcriptions = []
    
    if TRANSCRIPTIONS_DIR.exists():
        for file_path in TRANSCRIPTIONS_DIR.glob("*.txt"):
            if file_path.is_file():
                stat = file_path.stat()
                transcriptions.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime)
                })
    
    return transcriptions

@app.get("/transcriptions/{filename}")
async def get_transcription(filename: str):
    """Get transcription content by filename."""
    file_path = TRANSCRIPTIONS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    try:
        content = file_path.read_text(encoding='utf-8')
        stat = file_path.stat()
        
        return {
            "filename": filename,
            "content": content,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading transcription: {str(e)}")

@app.get("/transcriptions-list")
async def get_transcription_names():
    """Get list of available transcription file names only."""
    files = []
    if TRANSCRIPTIONS_DIR.exists():
        files = [f.name for f in TRANSCRIPTIONS_DIR.glob("*.txt") if f.is_file()]
    return {"files": sorted(files), "count": len(files)}

@app.post("/transcriptions/upload")
async def upload_audio_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload audio file for transcription."""
    if not TRANSCRIPTION_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Audio transcription not available. This feature requires additional audio processing dependencies."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file
    file_id = str(uuid4())
    file_extension = Path(file.filename).suffix
    saved_path = UPLOADS_DIR / f"{file_id}{file_extension}"
    
    try:
        content = await file.read()
        saved_path.write_bytes(content)
        
        # Add transcription task to background
        background_tasks.add_task(transcribe_audio_file, saved_path, file_id)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "File uploaded successfully. Transcription started.",
            "status": "processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def transcribe_audio_file(file_path: Path, file_id: str):
    """Background task to transcribe audio file."""
    if not TRANSCRIPTION_AVAILABLE:
        return
        
    try:
        transcriber = create_transcriber(model_name="base")
        result = await transcriber.transcribe_file(file_path)
        
        # Save transcription
        output_path = TRANSCRIPTIONS_DIR / f"{file_id}.txt"
        output_path.write_text(result.text, encoding='utf-8')
        
        # Clean up uploaded file
        file_path.unlink()
        
    except Exception as e:
        print(f"Transcription error for {file_id}: {e}")

# Analysis endpoints
@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    """Analyze transcribed text."""
    try:
        result = await analyze_transcription_tool(request.text, request.analysis_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/correct")
async def correct_text(request: CorrectionRequest):
    """Correct transcription errors."""
    try:
        result = await correct_transcription_tool(
            request.text, 
            request.context, 
            request.correction_level
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")

@app.post("/summarize")
async def summarize_file(request: SummarizationRequest):
    """Summarize transcription file."""
    try:
        result = await summarize_transcription_file_tool(
            request.filename,
            request.summary_type,
            request.extract_actions,
            request.extract_key_points,
            request.max_summary_length
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/translate")
async def translate_file(request: TranslationRequest):
    """Translate transcription file."""
    try:
        result = await translate_transcription_file_tool(
            request.filename,
            request.target_language,
            request.source_language
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: ChatMessage):
    """Chat with the ASR agent."""
    # Get available transcription files
    available_files = []
    if TRANSCRIPTIONS_DIR.exists():
        for file_path in TRANSCRIPTIONS_DIR.glob("*.txt"):
            if file_path.is_file():
                available_files.append(file_path.name)
    
    # Add transcription files context to the message if not already mentioned
    if available_files and "available transcription files" not in message.message.lower():
        files_info = f"\n\nAvailable transcription files: {', '.join(available_files[:50])}"
        if len(available_files) > 50:
            files_info += f" (and {len(available_files) - 50} more files)"
        message.message += files_info
    
    agent = await get_or_create_agent(message.session_id)
    response = await agent.answer_to(message.message)
    
    return ChatResponse(
        response=response.content,
        session_id=agent.session_id,
        tool_calls=[{
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments
        } for tc in response.tool_calls]
    )

@app.get("/sessions")
async def list_sessions():
    """List conversation sessions."""
    try:
        sessions = await history_manager.list_sessions()
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session."""
    try:
        # Close agent if active
        if session_id in active_agents:
            await active_agents[session_id].close()
            del active_agents[session_id]
        
        # Delete from database
        success = await history_manager.delete_session(session_id)
        
        if success:
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# Summaries and translations endpoints
@app.get("/summaries")
async def get_summaries(filename: Optional[str] = None, limit: int = 50):
    """Get transcription summaries."""
    try:
        summaries = await history_manager.get_transcription_summaries(filename=filename, limit=limit)
        return summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summaries: {str(e)}")

@app.get("/translations")
async def get_translations(filename: Optional[str] = None, target_language: Optional[str] = None, limit: int = 50):
    """Get transcription translations."""
    try:
        translations = await history_manager.get_transcription_translations(
            filename=filename, 
            target_language=target_language, 
            limit=limit
        )
        return translations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get translations: {str(e)}")

# Static file serving (for React build)
# @app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("../frontend/build/favicon.ico")

# Catch-all route for React Router (should be last)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serve React app for all non-API routes."""
    react_build_path = Path("../frontend/build")
    if react_build_path.exists():
        return FileResponse("../frontend/build/index.html")
    else:
        return {"message": "React frontend not built. Run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,  # Disable reload in production
        log_level="info"
    )
