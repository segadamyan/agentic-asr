"""Integration with the transcriber to create a real transcription tool."""
import os
from typing import Dict, Any

from ..core.models import ToolDefinition, LLMProviderConfig, History, Message, RoleEnum, MessageType
from ..llm.providers import create_llm_provider


async def analyze_transcription_tool(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """Advanced text analysis tool for transcribed content."""
    try:
        if analysis_type == "summary":
            sentences = text.split('. ')
            if len(sentences) <= 3:
                summary = text
            else:
                summary_sentences = [
                    sentences[0],
                    sentences[len(sentences) // 2],
                    sentences[-1]
                ]
                summary = '. '.join(summary_sentences)

            return {
                "analysis_type": "summary",
                "result": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if text else 0
            }

        elif analysis_type == "keywords":
            import re

            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'have', 'has', 'had', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'
            }

            filtered_words = [word for word in words if word not in stop_words]
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "analysis_type": "keywords",
                "keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
                "total_words": len(words),
                "unique_words": len(set(words)),
                "filtered_words": len(filtered_words)
            }

        elif analysis_type == "sentiment":
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
                'happy', 'joy', 'love', 'like', 'enjoy', 'pleased', 'satisfied', 'positive',
                'success', 'successful', 'perfect', 'brilliant', 'outstanding'
            }

            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
                'sad', 'angry', 'frustrated', 'disappointed', 'negative', 'wrong', 'problem',
                'issue', 'fail', 'failure', 'error', 'mistake', 'difficult'
            }

            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)

            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                sentiment = "neutral"
                confidence = 0.0
            else:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
                if sentiment_score > 0.2:
                    sentiment = "positive"
                elif sentiment_score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                confidence = abs(sentiment_score)

            return {
                "analysis_type": "sentiment",
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_words_found": positive_count,
                "negative_words_found": negative_count,
                "sentiment_score": sentiment_score if total_sentiment_words > 0 else 0
            }

        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}. Supported types: summary, keywords, sentiment"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }


async def correct_transcription_tool(
        original_text: str,
        context: str = "",
        correction_level: str = "medium",
) -> Dict[str, Any]:
    """Tool for correcting transcription errors using LLM intelligence."""
    try:
        correction_prompts = {
            "light": """Please review and lightly correct any obvious transcription errors in the following text. 
Focus only on clear spelling mistakes, missing punctuation, and obvious word recognition errors. 
Preserve the original meaning and style as much as possible.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations.""",

            "medium": """Please review and correct transcription errors in the following text.
Fix spelling mistakes, grammar issues, punctuation, and improve readability while maintaining the original meaning.
Correct obvious word recognition errors and ensure proper sentence structure.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations.""",

            "heavy": """Please thoroughly review and correct this transcription text.
Fix all spelling, grammar, and punctuation errors. Improve sentence structure and readability.
Correct word recognition errors and ensure the text flows naturally while preserving the original meaning and intent.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations."""
        }

        if correction_level not in correction_prompts:
            return {
                "success": False,
                "error": f"Invalid correction level: {correction_level}. Use 'light', 'medium', or 'heavy'"
            }

        context_section = f"\nContext: {context}" if context else ""

        prompt = correction_prompts[correction_level].format(
            text=original_text,
            context_section=context_section
        )

        llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        # Create LLM provider
        llm_provider = create_llm_provider(llm_config)

        history = History(session_id="correction_session")
        user_message = Message(
            role=RoleEnum.USER,
            content=prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)

        response = await llm_provider.generate_response(history=history)
        corrected_text = response.content.strip()

        # Calculate metrics
        original_length = len(original_text)
        corrected_length = len(corrected_text)

        # Simple word-level change detection
        original_words = original_text.split()
        corrected_words = corrected_text.split()

        changes_made = 0
        for i, word in enumerate(original_words):
            if i >= len(corrected_words) or word != corrected_words[i]:
                changes_made += 1
        changes_made += abs(len(original_words) - len(corrected_words))

        return {
            "success": True,
            "corrected_text": corrected_text,
            "original_text": original_text,
            "correction_level": correction_level,
            "context_used": context,
            "metrics": {
                "original_length": original_length,
                "corrected_length": corrected_length,
                "original_word_count": len(original_words),
                "corrected_word_count": len(corrected_words),
                "estimated_changes": changes_made,
                "change_percentage": (changes_made / len(original_words) * 100) if original_words else 0
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Transcription correction failed: {str(e)}"
        }


def get_enhanced_tools() -> list[ToolDefinition]:
    """Get enhanced tools including real transcription capabilities."""
    return [
        ToolDefinition(
            name="analyze_transcription",
            description="Perform advanced analysis on transcribed text",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The transcribed text to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: 'summary', 'keywords', or 'sentiment'",
                        "enum": ["summary", "keywords", "sentiment"],
                        "default": "summary"
                    }
                },
                "required": ["text"]
            },
            function=analyze_transcription_tool
        ),
        ToolDefinition(
            name="correct_transcription",
            description="Correct transcription errors using LLM intelligence",
            parameters={
                "type": "object",
                "properties": {
                    "original_text": {
                        "type": "string",
                        "description": "The original transcribed text with potential errors"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context to help with correction (e.g., topic, speaker details)"
                    },
                    "correction_level": {
                        "type": "string",
                        "description": "Level of correction to apply: 'light', 'medium', or 'heavy'",
                        "enum": ["light", "medium", "heavy"],
                        "default": "medium"
                    },
                },
                "required": ["original_text"]
            },
            function=correct_transcription_tool
        )
    ]
