"""LLM provider implementations for different services."""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import anthropic
import openai

from ..core.models import (
    History, Message, RoleEnum, MessageType, LLMProviderConfig,
    GenerationSettings, ToolCall
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_response(
            self,
            history: History,
            tools: Optional[List[Dict[str, Any]]] = None,
            settings: Optional[GenerationSettings] = None
    ) -> Message:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    async def generate_response(
            self,
            history: History,
            tools: Optional[List[Dict[str, Any]]] = None,
            settings: Optional[GenerationSettings] = None
    ) -> Message:
        """Generate response using OpenAI API."""
        try:
            messages = self._convert_history_to_openai(history)

            request_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": settings.temperature if settings else self.config.temperature,
                "max_tokens": settings.max_tokens if settings and settings.max_tokens else self.config.max_tokens,
                "top_p": settings.top_p if settings else self.config.top_p,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**request_params)

            return self._convert_openai_response(response)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return Message(
                role=RoleEnum.ASSISTANT,
                content=f"I apologize, but I encountered an error: {str(e)}",
                message_type=MessageType.TEXT
            )

    def _convert_history_to_openai(self, history: History) -> List[Dict[str, Any]]:
        """Convert our history format to OpenAI format."""
        openai_messages = []

        for msg in history.messages:
            if msg.role == RoleEnum.TOOL:
                for tool_result in msg.tool_call_results:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": json.dumps(
                            tool_result.result) if tool_result.success else f"Error: {tool_result.error}"
                    })
            else:
                openai_msg = {
                    "role": msg.role.value,
                    "content": msg.content
                }

                if msg.tool_calls:
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]

                openai_messages.append(openai_msg)

        return openai_messages

    def _convert_openai_response(self, response) -> Message:
        """Convert OpenAI response to our Message format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return Message(
            role=RoleEnum.ASSISTANT,
            content=message.content or "",
            message_type=MessageType.TOOL_CALL if tool_calls else MessageType.TEXT,
            tool_calls=tool_calls
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def generate_response(
            self,
            history: History,
            tools: Optional[List[Dict[str, Any]]] = None,
            settings: Optional[GenerationSettings] = None
    ) -> Message:
        """Generate response using Anthropic API."""
        try:
            messages, system_prompt = self._convert_history_to_anthropic(history)

            request_params = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": settings.max_tokens if settings and settings.max_tokens else (
                        self.config.max_tokens or 4000),
                "temperature": settings.temperature if settings else self.config.temperature,
                "top_p": settings.top_p if settings else self.config.top_p,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if tools:
                anthropic_tools = self._convert_tools_to_anthropic(tools)
                request_params["tools"] = anthropic_tools

            response = await self.client.messages.create(**request_params)

            return self._convert_anthropic_response(response)

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return Message(
                role=RoleEnum.ASSISTANT,
                content=f"I apologize, but I encountered an error: {str(e)}",
                message_type=MessageType.TEXT
            )

    def _convert_history_to_anthropic(self, history: History) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert our history format to Anthropic format."""
        anthropic_messages = []
        system_prompt = None

        for msg in history.messages:
            if msg.role == RoleEnum.SYSTEM:
                system_prompt = msg.content
            elif msg.role == RoleEnum.TOOL:
                for tool_result in msg.tool_call_results:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result.tool_call_id,
                                "content": json.dumps(
                                    tool_result.result) if tool_result.success else f"Error: {tool_result.error}"
                            }
                        ]
                    })
            else:
                content = []

                if msg.content:
                    content.append({
                        "type": "text",
                        "text": msg.content
                    })

                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments
                    })

                if content:
                    anthropic_messages.append({
                        "role": msg.role.value,
                        "content": content
                    })

        return anthropic_messages, system_prompt

    def _convert_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        return anthropic_tools

    def _convert_anthropic_response(self, response) -> Message:
        """Convert Anthropic response to our Message format."""
        content_text = ""
        tool_calls = []

        for content in response.content:
            if content.type == "text":
                content_text += content.text
            elif content.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=content.id,
                    name=content.name,
                    arguments=content.input
                ))

        return Message(
            role=RoleEnum.ASSISTANT,
            content=content_text,
            message_type=MessageType.TOOL_CALL if tool_calls else MessageType.TEXT,
            tool_calls=tool_calls
        )


def create_llm_provider(config: LLMProviderConfig) -> LLMProvider:
    """Factory function to create appropriate LLM provider."""
    if config.provider_name.lower() == "openai":
        return OpenAIProvider(config)
    elif config.provider_name.lower() == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider_name}")
