"""History management for persistent storage of conversations."""

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from .models import History, Message, RoleEnum

Base = declarative_base()


class ConversationRecord(Base):
    """Database model for conversation sessions."""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    session_metadata = Column(Text)  # JSON string


class MessageRecord(Base):
    """Database model for individual messages."""
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String, default="text")
    tool_calls = Column(Text)  # JSON string
    tool_call_results = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.now)
    message_metadata = Column(Text)  # JSON string


class TranscriptionSummaryRecord(Base):
    """Database model for transcription summaries."""
    __tablename__ = "transcription_summaries"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    summary_type = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    key_points = Column(Text)  # JSON string
    actions = Column(Text)  # JSON string
    summary_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class HistoryManager:
    """Manages conversation history with persistent storage."""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///./data/agentic_asr.db"):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        if "sqlite" in self.database_url and ":///" in self.database_url:
            db_path = self.database_url.split("///")[1]
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

    async def initialize(self):
        """Initialize the database connection and create tables."""
        self.engine = create_async_engine(self.database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession)
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_history(self, history: History) -> None:
        """Save conversation history to database."""
        if not self.async_session:
            await self.initialize()

        async with self.async_session() as session:
            conversation = ConversationRecord(
                id=history.session_id,
                session_id=history.session_id,
                created_at=history.created_at,
                session_metadata=json.dumps(history.metadata)
            )
            await session.merge(conversation)

            for message in history.messages:
                message_record = MessageRecord(
                    id=message.id,
                    session_id=history.session_id,
                    role=message.role.value,
                    content=message.content,
                    message_type=message.message_type.value,
                    tool_calls=json.dumps([tc.model_dump() for tc in message.tool_calls]),
                    tool_call_results=json.dumps([tr.model_dump() for tr in message.tool_call_results]),
                    timestamp=message.timestamp,
                    message_metadata=json.dumps(message.metadata)
                )
                await session.merge(message_record)

            await session.commit()

    async def load_history(self, session_id: str) -> Optional[History]:
        """Load conversation history from database."""
        if not self.async_session:
            await self.initialize()

        async with self.async_session() as session:
            conversation = await session.get(ConversationRecord, session_id)
            if not conversation:
                return None

            from sqlalchemy import select
            stmt = select(MessageRecord).where(MessageRecord.session_id == session_id).order_by(MessageRecord.timestamp)
            result = await session.execute(stmt)
            message_records = result.scalars().all()

            messages = []
            for record in message_records:
                tool_calls = []
                if record.tool_calls:
                    tool_calls_data = json.loads(record.tool_calls)
                    for tc_data in tool_calls_data:
                        from .models import ToolCall
                        tool_calls.append(ToolCall(**tc_data))
                
                tool_call_results = []
                if record.tool_call_results:
                    tool_results_data = json.loads(record.tool_call_results)
                    for tr_data in tool_results_data:
                        from .models import ToolResult
                        tool_call_results.append(ToolResult(**tr_data))
                
                message = Message(
                    id=record.id,
                    role=RoleEnum(record.role),
                    content=record.content,
                    message_type=record.message_type,
                    tool_calls=tool_calls,
                    tool_call_results=tool_call_results,
                    timestamp=record.timestamp,
                    metadata=json.loads(record.message_metadata) if record.message_metadata else {}
                )
                messages.append(message)

            history = History(
                messages=messages,
                session_id=conversation.session_id,
                created_at=conversation.created_at,
                metadata=json.loads(conversation.session_metadata) if conversation.session_metadata else {}
            )

            return history

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent conversation sessions."""
        if not self.async_session:
            await self.initialize()

        async with self.async_session() as session:
            from sqlalchemy import select, desc
            stmt = select(ConversationRecord).order_by(desc(ConversationRecord.updated_at)).limit(limit)
            result = await session.execute(stmt)
            conversations = result.scalars().all()

            return [
                {
                    "session_id": conv.session_id,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    "metadata": json.loads(conv.session_metadata) if conv.session_metadata else {}
                }
                for conv in conversations
            ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session and all its messages."""
        if not self.async_session:
            await self.initialize()

        async with self.async_session() as session:
            from sqlalchemy import delete
            await session.execute(delete(MessageRecord).where(MessageRecord.session_id == session_id))
            
            conversation = await session.get(ConversationRecord, session_id)
            if conversation:
                await session.delete(conversation)
                await session.commit()
                return True
            return False

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()

    async def save_transcription_summary(
        self, 
        filename: str,
        file_path: str,
        summary_type: str,
        summary: str,
        key_points: List[str] = None,
        actions: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save transcription summary to database."""
        if not self.async_session:
            await self.initialize()

        from uuid import uuid4
        summary_id = str(uuid4())
        
        async with self.async_session() as session:
            summary_record = TranscriptionSummaryRecord(
                id=summary_id,
                filename=filename,
                file_path=file_path,
                summary_type=summary_type,
                summary=summary,
                key_points=json.dumps(key_points or []),
                actions=json.dumps(actions or []),
                summary_metadata=json.dumps(metadata or {}),
                created_at=datetime.now()
            )
            session.add(summary_record)
            await session.commit()
            
        return summary_id

    async def get_transcription_summaries(self, filename: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transcription summaries from database."""
        if not self.async_session:
            await self.initialize()

        async with self.async_session() as session:
            from sqlalchemy import select, desc
            
            if filename:
                stmt = select(TranscriptionSummaryRecord).where(
                    TranscriptionSummaryRecord.filename == filename
                ).order_by(desc(TranscriptionSummaryRecord.created_at)).limit(limit)
            else:
                stmt = select(TranscriptionSummaryRecord).order_by(
                    desc(TranscriptionSummaryRecord.created_at)
                ).limit(limit)
            
            result = await session.execute(stmt)
            summaries = result.scalars().all()

            return [
                {
                    "id": summary.id,
                    "filename": summary.filename,
                    "file_path": summary.file_path,
                    "summary_type": summary.summary_type,
                    "summary": summary.summary,
                    "key_points": json.loads(summary.key_points) if summary.key_points else [],
                    "actions": json.loads(summary.actions) if summary.actions else [],
                    "metadata": json.loads(summary.summary_metadata) if summary.summary_metadata else {},
                    "created_at": summary.created_at.isoformat() if summary.created_at else None,
                    "updated_at": summary.updated_at.isoformat() if summary.updated_at else None
                }
                for summary in summaries
            ]
