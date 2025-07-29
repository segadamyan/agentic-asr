"""Simplified agent implementation for Agentic ASR."""

import uuid
import logging
import os
from typing import List, Optional, Dict, Any

from .models import (
    History, Message, RoleEnum, MessageType, LLMProviderConfig, 
    GenerationSettings, ToolDefinition, ToolCall
)
from .events import EventTopic, PhonyTopic
from .token_manager import TokenManager
from .tools import ToolInvoker, ToolExecutor, get_default_tools
from ..tools.enhanced import get_enhanced_tools
from .history import HistoryManager
from ..llm.providers import LLMProvider, create_llm_provider
from ..utils.logging import logger


class SimpleAgent:
    """
    Simplified agent for ASR processing with LLM interaction.
    
    This agent manages conversations with LLM providers, handles tool execution,
    and maintains conversation history within token limits.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_config: LLMProviderConfig,
        tools: Optional[List[ToolDefinition]] = None,
        max_tokens: int = 140000,
        recent_token_budget: int = 30000,
        topic: Optional[EventTopic] = None,
        history_manager: Optional[HistoryManager] = None,
        session_id: Optional[str] = None
    ):
        """Initialize the SimpleAgent.
        
        Args:
            name: Name of the agent
            system_prompt: System prompt for the LLM
            llm_config: Configuration for the LLM provider
            tools: List of available tools (defaults to ASR tools)
            max_tokens: Maximum tokens for conversation context
            recent_token_budget: Token budget for recent messages
            topic: Event topic for publishing events
            history_manager: Manager for persistent history storage
            session_id: Existing session ID to continue conversation
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.topic = topic or PhonyTopic()
        self.history_manager = history_manager
        
        self.tools = tools or get_enhanced_tools()
        
        self.token_manager = TokenManager(max_tokens, recent_token_budget)
        self.tool_invoker = ToolInvoker(self.tools)
        self.tool_executor = ToolExecutor(self.tool_invoker, self.topic, self.name)
        
        self.llm_provider = create_llm_provider(self.llm_config)
        
        self.session_id = session_id or str(uuid.uuid4())
        self.history = History(session_id=self.session_id)
        
        if system_prompt:
            system_message = Message(
                role=RoleEnum.SYSTEM,
                content=system_prompt,
                message_type=MessageType.TEXT
            )
            self.history.add_message(system_message)

    async def initialize(self) -> None:
        """Initialize the agent (load history if available)."""
        if self.history_manager and len(self.history.messages) <= 1:  # Only system message
            # Try to load existing history
            loaded_history = await self.history_manager.load_history(self.session_id)
            if loaded_history:
                self.history = loaded_history
                logger.info(f"Loaded history for session {self.session_id} with {len(self.history.messages)} messages")

    async def answer_to(self, query: str, settings: Optional[GenerationSettings] = None) -> Message:
        """
        Process a user query and return an AI response.
        
        Args:
            query: User input query
            settings: Optional generation settings
            
        Returns:
            Assistant's response message
        """
        await self._publish_start_event(query)
        
        # Add user message to history
        user_message = self.history.add_user_message(query)
        
        # Manage context window
        await self.token_manager.manage_context(self.history)
        
        logger.info(f"Processing query for {self.name}: {query[:100]}...")
        
        # Main conversation loop
        assistant_message = await self._conversation_loop(settings)
        
        # Save history if manager is available
        if self.history_manager:
            try:
                await self.history_manager.save_history(self.history)
            except Exception as e:
                logger.error(f"Failed to save history: {e}")
        
        await self._publish_end_event(assistant_message)
        return assistant_message

    async def _conversation_loop(self, settings: Optional[GenerationSettings] = None) -> Message:
        """Main conversation loop with tool execution."""
        max_iterations = 5
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get response from LLM
            response = await self.llm_provider.generate_response(
                history=self.history,
                tools=self.tool_invoker.get_tool_definitions_for_llm(),
                settings=settings
            )
            
            logger.debug(f"LLM response: {response.content[:200]}...")
            
            # If no tool calls, we're done
            if not response.tool_calls:
                self.history.add_message(response)
                return response
            
            # Add assistant message with tool calls
            self.history.add_message(response)
            
            # Execute tool calls
            tool_results = await self.tool_executor.invoke_tools(response)
            
            # Add tool results to history
            tool_message = Message(
                role=RoleEnum.TOOL,
                content="",
                message_type=MessageType.TOOL_RESULT,
                tool_call_results=tool_results
            )
            self.history.add_message(tool_message)
            
            # Check if all tools succeeded
            all_success = all(result.success for result in tool_results)
            if all_success:
                # Continue conversation to get final response
                continue
            else:
                # Some tools failed, but continue anyway
                logger.warning("Some tool calls failed, continuing conversation")
                continue
        
        # If we've exhausted iterations, return a fallback response
        fallback_response = Message(
            role=RoleEnum.ASSISTANT,
            content="I apologize, but I encountered some difficulties processing your request. Could you please try rephrasing your question?",
            message_type=MessageType.TEXT
        )
        self.history.add_message(fallback_response)
        return fallback_response

    async def clear_history(self) -> None:
        """Clear conversation history but keep system message."""
        system_messages = [msg for msg in self.history.messages if msg.role == RoleEnum.SYSTEM]
        self.history.clear()
        for msg in system_messages:
            self.history.add_message(msg)

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.history.messages),
            "token_usage": self.token_manager.get_token_usage(self.history),
            "last_message_time": self.history.messages[-1].timestamp if self.history.messages else None
        }

    async def export_conversation(self) -> Dict[str, Any]:
        """Export the full conversation for analysis or backup."""
        return {
            "session_id": self.session_id,
            "agent_name": self.name,
            "system_prompt": self.system_prompt,
            "created_at": self.history.created_at,
            "messages": [msg.model_dump() for msg in self.history.messages],
            "metadata": self.history.metadata
        }

    async def _publish_start_event(self, query: str) -> None:
        """Publish event when starting to process a query."""
        await self.topic.publish(
            event_type="agent.start_answer",
            source=self.name,
            data={"query": query, "session_id": self.session_id}
        )

    async def _publish_end_event(self, message: Message) -> None:
        """Publish event when finishing query processing."""
        await self.topic.publish(
            event_type="agent.new_message",
            source=self.name,
            data={"message": message.content, "session_id": self.session_id}
        )
        await self.topic.publish(
            event_type="agent.end_answer",
            source=self.name,
            data={"session_id": self.session_id}
        )

    def add_tool(self, tool: ToolDefinition) -> None:
        """Add a new tool to the agent."""
        self.tools.append(tool)
        self.tool_invoker = ToolInvoker(self.tools)
        self.tool_executor = ToolExecutor(self.tool_invoker, self.topic, self.name)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent."""
        original_count = len(self.tools)
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        
        if len(self.tools) < original_count:
            self.tool_invoker = ToolInvoker(self.tools)
            self.tool_executor = ToolExecutor(self.tool_invoker, self.topic, self.name)
            return True
        return False

    async def close(self) -> None:
        """Clean up resources."""
        if self.history_manager:
            await self.history_manager.close()


async def create_asr_agent(
    name: str = "ASR-Agent",
    system_prompt: str = "You are an intelligent ASR assistant that helps process and analyze transcribed speech. You can transcribe audio, summarize text, and extract keywords.",
    llm_config: Optional[LLMProviderConfig] = None,
    database_url: str = "sqlite+aiosqlite:///./data/agentic_asr.db",
    session_id: Optional[str] = None
) -> SimpleAgent:
    """Create and initialize an ASR agent with default settings."""
    
    if not llm_config:
        llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            temperature=0.7
        )
    
    history_manager = HistoryManager(database_url)
    await history_manager.initialize()
    
    agent = SimpleAgent(
        name=name,
        system_prompt=system_prompt,
        llm_config=llm_config,
        history_manager=history_manager,
        session_id=session_id
    )
    
    await agent.initialize()
    
    transcriptions_dir = "./data/transcriptions"
    if os.path.exists(transcriptions_dir):
        transcription_files = [f for f in os.listdir(transcriptions_dir) 
                              if os.path.isfile(os.path.join(transcriptions_dir, f))]
        
        if transcription_files:
            files_list = "\n".join([f"- {file}" for file in transcription_files])
            transcription_message = f"""I have access to the following transcription files in the data/transcriptions folder:

{files_list}

I can help you analyze, summarize, or work with any of these transcriptions. Just let me know what you'd like to do!"""
            
            user_message = Message(
                role=RoleEnum.USER,
                content=transcription_message,
                message_type=MessageType.TEXT
            )
            agent.history.add_message(user_message)
    
    return agent
