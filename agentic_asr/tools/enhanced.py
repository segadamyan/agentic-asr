"""Integration with the transcriber to create a real transcription tool."""
import os
from pathlib import Path
from typing import Dict, Any

from ..core.history import HistoryManager
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


async def summarize_transcription_file_tool(
    filename: str,
    summary_type: str = "comprehensive",
    extract_actions: bool = True,
    extract_key_points: bool = True,
    max_summary_length: int = 500
) -> Dict[str, Any]:
    """Tool for comprehensive summarization of transcription files with key points and action extraction."""
    try:
        transcription_paths = [
            Path("data/transcriptions"),
        ]
        
        file_path = None
        for base_path in transcription_paths:
            potential_path = base_path / filename
            if potential_path.exists():
                file_path = potential_path
                break
            
            for ext in ['.txt', '.json', '.md']:
                if not filename.endswith(ext):
                    potential_path = base_path / f"{filename}{ext}"
                    if potential_path.exists():
                        file_path = potential_path
                        break
            if file_path:
                break
        
        if not file_path:
            return {
                "success": False,
                "error": f"Transcription file '{filename}' not found in any of the transcription directories: {[str(p) for p in transcription_paths]}"
            }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}"
            }
        
        if not content:
            return {
                "success": False,
                "error": "File is empty or contains no readable content"
            }

        summary_prompts = {
            "brief": """Please provide a brief summary of the following transcription in 2-3 sentences:

{content}

Provide only the summary without explanations.""",

            "detailed": """Please provide a detailed summary of the following transcription, covering all main topics and important details:

{content}

Provide only the summary without explanations.""",

            "comprehensive": """Please analyze the following transcription and provide:
1. A comprehensive summary
2. Key points (bullet format)
3. Action items or tasks mentioned (if any)

Transcription:
{content}

Format your response as:
SUMMARY:
[Your summary here]

KEY POINTS:
• [Point 1]
• [Point 2]
• [Point 3]
...

ACTIONS:
• [Action 1]
• [Action 2]
...
(If no actions are mentioned, write "No specific actions identified")"""
        }

        if summary_type not in summary_prompts:
            return {
                "success": False,
                "error": f"Invalid summary type: {summary_type}. Use 'brief', 'detailed', or 'comprehensive'"
            }

        prompt = summary_prompts[summary_type].format(content=content)

        llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        llm_provider = create_llm_provider(llm_config)
        
        history = History(session_id="summarization_session")
        user_message = Message(
            role=RoleEnum.USER,
            content=prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)

        response = await llm_provider.generate_response(history=history)
        llm_output = response.content.strip()

        if summary_type == "comprehensive":
            summary = ""
            key_points = []
            actions = []
            
            sections = llm_output.split("\n\n")
            current_section = ""
            
            for section in sections:
                section = section.strip()
                if section.startswith("SUMMARY:"):
                    current_section = "summary"
                    summary = section.replace("SUMMARY:", "").strip()
                elif section.startswith("KEY POINTS:"):
                    current_section = "key_points"
                elif section.startswith("ACTIONS:"):
                    current_section = "actions"
                elif current_section == "summary" and summary:
                    summary += " " + section
                elif current_section == "key_points":
                    lines = section.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("•") or line.startswith("-"):
                            key_points.append(line.lstrip("•-").strip())
                elif current_section == "actions":
                    lines = section.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("•") or line.startswith("-"):
                            action_text = line.lstrip("•-").strip()
                            if not action_text.lower().startswith("no specific actions"):
                                actions.append(action_text)

            result = {
                "success": True,
                "file_path": str(file_path),
                "filename": filename,
                "summary_type": summary_type,
                "summary": summary,
                "key_points": key_points if extract_key_points else [],
                "actions": actions if extract_actions else [],
                "metadata": {
                    "original_length": len(content),
                    "summary_length": len(summary),
                    "compression_ratio": len(summary) / len(content) if content else 0,
                    "key_points_count": len(key_points),
                    "actions_count": len(actions),
                    "file_size_bytes": file_path.stat().st_size
                }
            }

            try:
                history_manager = HistoryManager()
                summary_id = await history_manager.save_transcription_summary(
                    filename=filename,
                    file_path=str(file_path),
                    summary_type=summary_type,
                    summary=summary,
                    key_points=key_points if extract_key_points else [],
                    actions=actions if extract_actions else [],
                    metadata=result["metadata"]
                )
                result["database_id"] = summary_id
                result["saved_to_database"] = True
                await history_manager.close()
            except Exception as db_error:
                result["database_error"] = str(db_error)
                result["saved_to_database"] = False
        else:
            # For brief and detailed summaries
            result = {
                "success": True,
                "file_path": str(file_path),
                "filename": filename,
                "summary_type": summary_type,
                "summary": llm_output,
                "metadata": {
                    "original_length": len(content),
                    "summary_length": len(llm_output),
                    "compression_ratio": len(llm_output) / len(content) if content else 0,
                    "file_size_bytes": file_path.stat().st_size
                }
            }

            # Save to database
            try:
                history_manager = HistoryManager()
                summary_id = await history_manager.save_transcription_summary(
                    filename=filename,
                    file_path=str(file_path),
                    summary_type=summary_type,
                    summary=llm_output,
                    key_points=[],
                    actions=[],
                    metadata=result["metadata"]
                )
                result["database_id"] = summary_id
                result["saved_to_database"] = True
                await history_manager.close()
            except Exception as db_error:
                result["database_error"] = str(db_error)
                result["saved_to_database"] = False

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Summarization failed: {str(e)}"
        }


async def get_transcription_summaries_tool(
    filename: str = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Tool to retrieve saved transcription summaries from database."""
    try:
        history_manager = HistoryManager()
        summaries = await history_manager.get_transcription_summaries(filename=filename, limit=limit)
        await history_manager.close()
        
        return {
            "success": True,
            "summaries": summaries,
            "count": len(summaries),
            "filename_filter": filename
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to retrieve summaries: {str(e)}"
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
        ),
        ToolDefinition(
            name="summarize_transcription_file",
            description="Summarize transcription files with key points and action extraction",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to summarize (will search in transcription directories)"
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summary to generate: 'brief', 'detailed', or 'comprehensive'",
                        "enum": ["brief", "detailed", "comprehensive"],
                        "default": "comprehensive"
                    },
                    "extract_actions": {
                        "type": "boolean",
                        "description": "Whether to extract action items from the transcription",
                        "default": True
                    },
                    "extract_key_points": {
                        "type": "boolean",
                        "description": "Whether to extract key points from the transcription",
                        "default": True
                    },
                    "max_summary_length": {
                        "type": "integer",
                        "description": "Maximum length of the summary",
                        "default": 500
                    }
                },
                "required": ["filename"]
            },
            function=summarize_transcription_file_tool
        ),
        ToolDefinition(
            name="get_transcription_summaries",
            description="Retrieve saved transcription summaries from database",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter to get summaries for a specific file"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of summaries to retrieve",
                        "default": 10
                    }
                },
                "required": []
            },
            function=get_transcription_summaries_tool
        )
    ]
