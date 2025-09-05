"""Tool execution and management."""

import asyncio
import logging
from typing import List, Dict, Any

from .events import EventTopic
from .models import ToolCall, ToolResult, ToolDefinition, Message

logger = logging.getLogger(__name__)


class ToolInvoker:
    """Handles tool invocation and execution."""

    def __init__(
            self,
            tools: List[ToolDefinition],
            system_settings: Dict[str, str] = None,
            handled_exceptions: tuple = ()
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.system_settings = system_settings or {}
        self.handled_exceptions = handled_exceptions

    def get_tool_definitions_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool definitions in format suitable for LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]

    async def invoke_tool(self, tool_call: ToolCall) -> ToolResult:
        """Invoke a single tool."""
        try:
            if tool_call.name not in self.tools:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=f"Unknown tool: {tool_call.name}"
                )

            tool = self.tools[tool_call.name]
            if not tool.function:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=f"Tool {tool_call.name} has no associated function"
                )

            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**tool_call.arguments)
            else:
                result = tool.function(**tool_call.arguments)

            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                result=result
            )

        except self.handled_exceptions as e:
            logger.warning(f"Handled exception in tool {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unhandled exception in tool {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )


class ToolExecutor:
    """Manages tool execution with event publishing."""

    def __init__(self, invoker: ToolInvoker, topic: EventTopic, agent_name: str):
        self.invoker = invoker
        self.topic = topic
        self.agent_name = agent_name

    async def invoke_tools(self, message: Message) -> List[ToolResult]:
        """Invoke all tools in a message sequentially."""
        results = []
        for tool_call in message.tool_calls:
            result = await self.invoke_single_tool(tool_call)
            results.append(result)
        return results

    async def invoke_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Invoke a single tool with event publishing."""
        await self.topic.publish(
            event_type="tool.start",
            source=self.agent_name,
            data={"tool_name": tool_call.name, "tool_call_id": tool_call.id}
        )

        result = await self.invoker.invoke_tool(tool_call)

        await self.topic.publish(
            event_type="tool.complete",
            source=self.agent_name,
            data={
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "success": result.success,
                "error": result.error
            }
        )

        return result


async def summarize_text_tool(text: str, max_length: int = 200) -> Dict[str, Any]:
    """Tool for summarizing transcribed text."""
    sentences = text.split('. ')
    summary = '. '.join(sentences[:3])
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary)
    }


async def extract_keywords_tool(text: str, max_keywords: int = 10) -> Dict[str, Any]:
    """Tool for extracting keywords from text."""
    words = text.lower().split()
    # Remove common words (basic stop words)
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    keywords = [word for word in words if word not in stop_words and len(word) > 3]

    keyword_counts = {}
    for word in keywords:
        keyword_counts[word] = keyword_counts.get(word, 0) + 1

    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:max_keywords]

    return {
        "keywords": [{"word": word, "frequency": count} for word, count in top_keywords],
        "total_words": len(words),
        "unique_words": len(set(words))
    }


def get_default_tools() -> List[ToolDefinition]:
    """Get default tools for content processing."""
    return [
        ToolDefinition(
            name="summarize_text",
            description="Summarize transcribed text",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_length": {"type": "integer", "description": "Maximum length of summary", "default": 200}
                },
                "required": ["text"]
            },
            function=summarize_text_tool
        ),
        ToolDefinition(
            name="extract_keywords",
            description="Extract keywords from transcribed text",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to extract keywords from"},
                    "max_keywords": {"type": "integer", "description": "Maximum number of keywords", "default": 10}
                },
                "required": ["text"]
            },
            function=extract_keywords_tool
        )
    ]


def get_enhanced_default_tools() -> List[ToolDefinition]:
    """Get enhanced tools including real transcription capabilities."""
    try:
        from ..tools.enhanced import get_enhanced_tools
        return get_enhanced_tools()
    except ImportError:
        return get_default_tools()
