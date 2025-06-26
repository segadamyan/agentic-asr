"""Integration with the transcriber to create a real transcription tool."""

from pathlib import Path
from typing import Dict, Any

from ..asr.transcriber import WhisperTranscriber
from ..core.models import ToolDefinition


async def real_transcribe_audio_tool(file_path: str, language: str = "auto", model: str = "base") -> Dict[str, Any]:
    """Tool for transcribing audio files using Whisper."""
    try:
        audio_path = Path(file_path)
        if not audio_path.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {file_path}"
            }

        transcriber = WhisperTranscriber(model_name=model)

        result = await transcriber.transcribe_file(
            audio_path,
            language=language if language != "auto" else None
        )

        return {
            "success": True,
            "transcription": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration": result.metadata.get("audio_duration"),
            "segments_count": result.metadata.get("segments_count"),
            "model_used": model
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }


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


def get_enhanced_tools() -> list[ToolDefinition]:
    """Get enhanced tools including real transcription capabilities."""
    return [
        ToolDefinition(
            name="transcribe_audio",
            description="Transcribe an audio file to text using Whisper AI",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file to transcribe"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for transcription (e.g., 'en', 'es', 'fr') or 'auto' for auto-detection",
                        "default": "auto"
                    },
                    "model": {
                        "type": "string",
                        "description": "Whisper model to use (tiny, base, small, medium, large)",
                        "default": "base"
                    }
                },
                "required": ["file_path"]
            },
            function=real_transcribe_audio_tool
        ),
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
        )
    ]
