"""Token management for conversation history."""

import logging
from typing import Optional

import tiktoken

from .models import History, Message

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages token usage and context window for conversations."""

    def __init__(self, max_tokens: int = 140000, recent_token_budget: int = 30000):
        self.max_tokens = max_tokens
        self.recent_token_budget = recent_token_budget
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}, using character approximation")
            return len(text) // 4

    def count_message_tokens(self, message: Message) -> int:
        """Count tokens in a message."""
        total = self.count_tokens(message.content)

        for tool_call in message.tool_calls:
            total += self.count_tokens(f"{tool_call.name}: {str(tool_call.arguments)}")

        for tool_result in message.tool_call_results:
            total += self.count_tokens(str(tool_result.result))

        return total

    def count_history_tokens(self, history: History) -> int:
        """Count total tokens in conversation history."""
        total = 0
        for message in history.messages:
            total += self.count_message_tokens(message)
        return total

    def get_token_usage(self, history: History) -> dict:
        """Get token usage statistics."""
        total_tokens = self.count_history_tokens(history)
        recent_messages = history.get_recent_messages(10)  # Last 10 messages
        recent_tokens = sum(self.count_message_tokens(msg) for msg in recent_messages)

        return {
            "total_tokens": total_tokens,
            "recent_tokens": recent_tokens,
            "max_tokens": self.max_tokens,
            "token_budget_remaining": max(0, self.max_tokens - total_tokens),
            "recent_budget_remaining": max(0, self.recent_token_budget - recent_tokens),
            "message_count": len(history.messages)
        }

    async def manage_context(self, history: History) -> None:
        """Manage context window by removing old messages if needed."""
        total_tokens = self.count_history_tokens(history)

        if total_tokens <= self.max_tokens:
            return

        logger.info(f"Context window exceeded ({total_tokens} > {self.max_tokens}), managing context")

        system_messages = [msg for msg in history.messages if msg.role.value == "system"]
        other_messages = [msg for msg in history.messages if msg.role.value != "system"]

        system_tokens = sum(self.count_message_tokens(msg) for msg in system_messages)
        available_tokens = self.max_tokens - system_tokens - 1000  # Reserve 1000 tokens for response

        kept_messages = []
        current_tokens = 0

        for message in reversed(other_messages):
            message_tokens = self.count_message_tokens(message)
            if current_tokens + message_tokens <= available_tokens:
                kept_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                break

        history.messages = system_messages + kept_messages

        new_total = self.count_history_tokens(history)
        removed_count = len(other_messages) - len(kept_messages)

        logger.info(f"Removed {removed_count} messages, new token count: {new_total}")

    def estimate_response_tokens(self, max_response_tokens: Optional[int] = None) -> int:
        """Estimate tokens needed for response."""
        if max_response_tokens:
            return min(max_response_tokens, 4000)  # Cap at 4000 tokens
        return 1000  # Default estimate
