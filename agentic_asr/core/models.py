"""Core data models for the agentic ASR system."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(str, Enum):
    """Types of messages."""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AUDIO_TRANSCRIPTION = "audio_transcription"


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool_call_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class Message(BaseModel):
    """A message in the conversation history."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: RoleEnum
    content: str
    message_type: MessageType = MessageType.TEXT
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_call_results: List[ToolResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return super().model_dump_json(**kwargs)


class History(BaseModel):
    """Conversation history container."""
    messages: List[Message] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a user message to the history."""
        message = Message(
            role=RoleEnum.USER,
            content=content,
            metadata=metadata or {}
        )
        self.add_message(message)
        return message

    def add_assistant_message(self, content: str, tool_calls: Optional[List[ToolCall]] = None) -> Message:
        """Add an assistant message to the history."""
        message = Message(
            role=RoleEnum.ASSISTANT,
            content=content,
            tool_calls=tool_calls or []
        )
        self.add_message(message)
        return message

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def get_recent_messages(self, count: int) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if count < len(self.messages) else self.messages


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    provider_name: str  # "openai", "anthropic", etc.
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0


class GenerationSettings(BaseModel):
    """Settings for text generation."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the agent."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    function: Optional[Any] = None  # The actual callable function


class ASRResult(BaseModel):
    """Result from speech recognition."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
