"""Integration with the transcriber to create a real transcription tool."""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import os
import re
import time
import logging

from ..core.history import HistoryManager
from ..core.models import ToolDefinition, LLMProviderConfig, History, Message, RoleEnum, MessageType
from ..llm.providers import create_llm_provider
from ..utils.logging import logger

# Tool-specific logger
tool_logger = logging.getLogger("agentic_asr.tools")


def log_tool_call(tool_name: str):
    """Decorator to log tool calls with execution time and parameters."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log the tool call start
            tool_logger.info(f"🔧 TOOL CALL START: {tool_name}")
            tool_logger.debug(f"Tool: {tool_name} | Args: {args} | Kwargs: {kwargs}")
            
            try:
                # Execute the tool
                result = await func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log success
                success = result.get("success", False) if isinstance(result, dict) else True
                status = "✅ SUCCESS" if success else "❌ FAILED"
                
                tool_logger.info(f"🔧 TOOL CALL END: {tool_name} | {status} | Duration: {execution_time:.2f}s")
                
                if isinstance(result, dict):
                    if "error" in result:
                        tool_logger.warning(f"Tool {tool_name} error: {result['error']}")
                    
                    # Log key result metrics
                    if "metadata" in result:
                        metadata = result["metadata"]
                        if "original_length" in metadata and "summary_length" in metadata:
                            tool_logger.debug(f"Tool {tool_name} processed {metadata['original_length']} chars -> {metadata['summary_length']} chars")
                    
                    if "total_results" in result:
                        tool_logger.debug(f"Tool {tool_name} returned {result['total_results']} results")
                    
                    if "files_count" in result:
                        tool_logger.debug(f"Tool {tool_name} processed {result['files_count']} files")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                tool_logger.error(f"🔧 TOOL CALL ERROR: {tool_name} | Duration: {execution_time:.2f}s | Error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Tool execution failed: {str(e)}",
                    "tool_name": tool_name,
                    "execution_time": execution_time
                }
        
        return wrapper
    return decorator


# Pydantic models for structured outputs
class TranscriptionSummary(BaseModel):
    """Structured output model for transcription summarization."""
    summary: str
    key_points: List[str]
    actions: List[str]
    language_detected: str


@log_tool_call("analyze_transcription")
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


@log_tool_call("correct_transcription")
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
Preserve the original meaning, style, and LANGUAGE as much as possible. Keep the corrected text in the same language as the original.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations, maintaining the same language as the original.""",

            "medium": """Please review and correct transcription errors in the following text.
Fix spelling mistakes, grammar issues, punctuation, and improve readability while maintaining the original meaning and LANGUAGE.
Correct obvious word recognition errors and ensure proper sentence structure. Keep the corrected text in the same language as the original.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations, maintaining the same language as the original.""",

            "heavy": """Please thoroughly review and correct this transcription text.
Fix all spelling, grammar, and punctuation errors. Improve sentence structure and readability.
Correct word recognition errors and ensure the text flows naturally while preserving the original meaning, intent, and LANGUAGE.
Keep the corrected text in the same language as the original transcription.

Original transcription: {text}
{context_section}

Please provide only the corrected text without explanations, maintaining the same language as the original."""
        }

        if correction_level not in correction_prompts:
            return {
                "success": False,
                "error": f"Invalid correction level: {correction_level}. Use 'light', 'medium', or 'heavy'"
            }

        context_section = f"\nContext: {context}" if context else ""

        llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        # Create LLM provider
        llm_provider = create_llm_provider(llm_config)

        # Check if content needs chunking for correction
        estimated_tokens = len(original_text) // 4  # Rough estimate: 1 token ≈ 4 characters

        if estimated_tokens > 100000:  # If content is too large, chunk it
            chunks = chunk_text(original_text, max_tokens=25000)  # Smaller chunks for correction
            
            # Correct each chunk
            corrected_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = correction_prompts[correction_level].format(
                    text=chunk,
                    context_section=context_section
                )
                
                history = History(session_id=f"correction_session_chunk_{i}")
                user_message = Message(
                    role=RoleEnum.USER,
                    content=chunk_prompt,
                    message_type=MessageType.TEXT
                )
                history.add_message(user_message)

                try:
                    response = await llm_provider.generate_response(history=history)
                    corrected_chunks.append(response.content.strip())
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to correct chunk {i+1}/{len(chunks)}: {str(e)}"
                    }
            
            # Combine corrected chunks with better joining
            corrected_text = " ".join(corrected_chunks)  # Use space instead of double newlines for corrections
        else:
            # Content is small enough to process directly
            prompt = correction_prompts[correction_level].format(
                text=original_text,
                context_section=context_section
            )

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


def chunk_text(text: str, max_tokens: int = 30000) -> list[str]:
    """
    Split text into chunks that fit within token limits.
    Uses approximate token counting (1 token ≈ 4 characters for safety).
    Handles various text formats including single-line transcriptions.
    """
    # Approximate max characters per chunk (conservative estimate)
    max_chars = max_tokens * 3  # Being conservative with token estimation
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Strategy 1: Try splitting by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    # If we have meaningful paragraphs (more than 1 and not just the whole text)
    if len(paragraphs) > 1 and len(paragraphs[0]) < len(text) * 0.9:
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, need to split it further
                    para_chunks = _split_large_text(paragraph, max_chars)
                    chunks.extend(para_chunks)
                    current_chunk = ""
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    else:
        # Strategy 2: Single line or no meaningful paragraphs - split by other methods
        current_chunk = text
    
    # If we still have a large chunk, split it
    if current_chunk and len(current_chunk) > max_chars:
        final_chunks = _split_large_text(current_chunk, max_chars)
        chunks.extend(final_chunks)
    elif current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_large_text(text: str, max_chars: int) -> list[str]:
    """
    Helper function to split large text using multiple strategies.
    """
    chunks = []
    current_chunk = ""
    
    # Strategy 1: Split by sentences (periods followed by space and capital letter or end)
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZԱ-Ֆա-և])', text)
    
    if len(sentences) > 1:
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by other methods
                    sentence_chunks = _split_by_punctuation(sentence, max_chars)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
            else:
                current_chunk += " " + sentence if current_chunk else sentence
    else:
        # Strategy 2: If no sentences, split by other punctuation
        current_chunk = text
    
    # If we still have a large chunk, split by punctuation
    if current_chunk and len(current_chunk) > max_chars:
        punct_chunks = _split_by_punctuation(current_chunk, max_chars)
        chunks.extend(punct_chunks)
    elif current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_by_punctuation(text: str, max_chars: int) -> list[str]:
    """
    Split text by various punctuation marks when sentences don't work.
    """
    chunks = []
    current_chunk = ""
    
    # Split by commas, semicolons, colons, and other punctuation
    import re
    segments = re.split(r'([,;:])\s*', text)
    
    if len(segments) > 1:
        for i in range(0, len(segments), 2):
            segment = segments[i]
            punct = segments[i + 1] if i + 1 < len(segments) else ""
            full_segment = segment + punct
            
            if len(current_chunk) + len(full_segment) > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = full_segment
                else:
                    # Even single segment is too long, split by words
                    word_chunks = _split_by_words(full_segment, max_chars)
                    chunks.extend(word_chunks)
                    current_chunk = ""
            else:
                current_chunk += full_segment
    else:
        # No punctuation, split by words as last resort
        current_chunk = text
    
    # If we still have a large chunk, split by words
    if current_chunk and len(current_chunk) > max_chars:
        word_chunks = _split_by_words(current_chunk, max_chars)
        chunks.extend(word_chunks)
    elif current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_by_words(text: str, max_chars: int) -> list[str]:
    """
    Split text by words as the last resort.
    """
    chunks = []
    current_chunk = ""
    words = text.split()
    
    for word in words:
        if len(current_chunk) + len(word) + 1 > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # Single word is extremely long (rare), just add it
                chunks.append(word)
                current_chunk = ""
        else:
            current_chunk += " " + word if current_chunk else word
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


@log_tool_call("summarize_transcription_file")
async def summarize_transcription_file_tool(
    filename: str,
    summary_type: str = "comprehensive",
    extract_actions: bool = True,
    extract_key_points: bool = True,
    max_summary_length: int = 500
) -> Dict[str, Any]:
    """Tool for comprehensive summarization of transcription files with key points and action extraction."""
    transcription_paths = [
        Path("../data/transcriptions"),  # From api directory
        Path("data/transcriptions"),      # From project root
        Path("/Users/sadamyan/workdir/agentic-asr/data/transcriptions"),  # Absolute path
    ]
    
    file_path = None
    for base_path in transcription_paths:
        # First try exact match
        potential_path = base_path / filename
        if potential_path.exists():
            file_path = potential_path
            break
        
        # Try with different extensions if not already has one
        for ext in ['.txt', '.json', '.md']:
            if not filename.endswith(ext):
                potential_path = base_path / f"{filename}{ext}"
                if potential_path.exists():
                    file_path = potential_path
                    break
        if file_path:
            break
            
        # Try fuzzy matching for common variations (spaces, case, etc.)
        if not file_path and base_path.exists():
            import re
            # Create a pattern that allows for flexible matching
            normalized_filename = re.sub(r'[^\w]', '', filename.lower())
            for file_candidate in base_path.glob('*'):
                if file_candidate.is_file():
                    candidate_normalized = re.sub(r'[^\w]', '', file_candidate.stem.lower())
                    if normalized_filename in candidate_normalized or candidate_normalized in normalized_filename:
                        # Additional check for common patterns like "Rearrange #261" vs "Rearrange#261"
                        if abs(len(normalized_filename) - len(candidate_normalized)) <= 2:
                            file_path = file_candidate
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

    # Debug info: print content stats
    print(f"File successfully found: {file_path}")
    print(f"Content length: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}...")
    print(f"Summary type: {summary_type}, Extract actions: {extract_actions}, Extract key points: {extract_key_points}, Max summary length: {max_summary_length}")

    summary_prompts = {
        "brief": """Please analyze the language of the following transcription and provide a brief summary in the SAME LANGUAGE as the original transcription. If the transcription is in Armenian, summarize in Armenian. If it's in English, summarize in English, etc.

Provide a brief summary in 2-3 sentences in the same language as the transcription:

{content}

Provide only the summary without explanations, and ensure it's in the same language as the original text.""",

        "detailed": """Please analyze the language of the following transcription and provide a detailed summary in the SAME LANGUAGE as the original transcription. If the transcription is in Armenian, summarize in Armenian. If it's in English, summarize in English, etc.

Provide a detailed summary covering all main topics and important details in the same language as the transcription:

{content}

Provide only the summary without explanations, and ensure it's in the same language as the original text.""",

        "comprehensive": """Please analyze the following transcription and provide a comprehensive analysis in the SAME LANGUAGE as the original transcription.

Transcription:
{content}

Please provide:
1. A comprehensive summary in the same language as the transcription
2. Key points extracted from the content (as a list)
3. Action items or tasks mentioned (as a list, empty if none)
4. The detected language of the transcription

Ensure all text content (summary, key points, actions) is in the same language as the original transcription."""
    }

    if summary_type not in summary_prompts:
        return {
            "success": False,
            "error": f"Invalid summary type: {summary_type}. Use 'brief', 'detailed', or 'comprehensive'"
        }

    # Check if content needs chunking (estimate tokens)
    estimated_tokens = len(content) // 4  # Rough estimate: 1 token ≈ 4 characters
    
    print(f"Esimated tokens: {estimated_tokens}, Max summary length: {max_summary_length}")

    llm_config = LLMProviderConfig(
        provider_name="openai",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    llm_provider = create_llm_provider(llm_config)

    # For comprehensive summaries, we'll use structured outputs
    use_structured_output = (summary_type == "comprehensive")

    if estimated_tokens > 100000:  # If content is too large, chunk it
        chunks = chunk_text(content, max_tokens=25000)  # Slightly smaller chunks for better processing
        
        print(f"Content is large ({estimated_tokens} estimated tokens), splitting into {len(chunks)} chunks")
        
        # Process each chunk and collect summaries
        chunk_summaries = []
        chunk_structured_data = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
            chunk_prompt = summary_prompts[summary_type].format(content=chunk)
            
            history = History(session_id=f"summarization_session_chunk_{i}")
            user_message = Message(
                role=RoleEnum.USER,
                content=chunk_prompt,
                message_type=MessageType.TEXT
            )
            history.add_message(user_message)

            try:
                if use_structured_output:
                    # Use structured output for comprehensive summaries
                    response = await llm_provider.generate_structured_response(
                        history=history,
                        response_format=TranscriptionSummary
                    )
                    # Extract structured content from the response
                    if hasattr(response, '_structured_content'):
                        chunk_structured_data.append(response._structured_content)
                    else:
                        # Fallback: parse from JSON content
                        import json
                        content_dict = json.loads(response.content)
                        chunk_structured_data.append(TranscriptionSummary(**content_dict))
                    print(f"Successfully processed chunk {i+1} with structured output")
                else:
                    response = await llm_provider.generate_response(history=history)
                    chunk_summaries.append(response.content.strip())
                    print(f"Successfully processed chunk {i+1}")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to process chunk {i+1}/{len(chunks)}: {str(e)}"
                }
        
        if use_structured_output:
            # Combine structured data from chunks
            all_summaries = [chunk.summary for chunk in chunk_structured_data]
            all_key_points = []
            all_actions = []
            detected_language = chunk_structured_data[0].language_detected if chunk_structured_data else "unknown"
            
            for chunk_data in chunk_structured_data:
                all_key_points.extend(chunk_data.key_points)
                all_actions.extend(chunk_data.actions)
            
            # Create a final consolidated summary using structured output
            combined_text = "\n\n".join(all_summaries)
            final_prompt = f"""Please analyze the language of the following chunk summaries and provide a final consolidated analysis in the SAME LANGUAGE as the original transcription.

These are summaries of different parts of a large transcription. Please create a coherent final analysis:

{combined_text}

Key points from chunks: {'; '.join(all_key_points[:20])}
Actions from chunks: {'; '.join(all_actions[:10])}

Please provide a final consolidated summary, refined key points, and consolidated actions all in the same language as the original transcription."""
            
            final_history = History(session_id="final_summarization_session")
            final_message = Message(
                role=RoleEnum.USER,
                content=final_prompt,
                message_type=MessageType.TEXT
            )
            final_history.add_message(final_message)

            final_response = await llm_provider.generate_structured_response(
                history=final_history,
                response_format=TranscriptionSummary
            )
            # Extract structured content from the response
            if hasattr(final_response, '_structured_content'):
                structured_result = final_response._structured_content
            else:
                # Fallback: parse from JSON content
                import json
                content_dict = json.loads(final_response.content)
                structured_result = TranscriptionSummary(**content_dict)
        else:
            # Now summarize the chunk summaries for non-comprehensive
            combined_summaries = "\n\n".join(chunk_summaries)
            
            final_prompt = f"""Please analyze the language of the following chunk summaries and provide a final consolidated summary in the SAME LANGUAGE as the original transcription.

These are summaries of different parts of a large transcription. Please create a coherent final summary in the same language:

{combined_summaries}

Please provide a final consolidated summary in the same language as the original transcription."""
            
            final_history = History(session_id="final_summarization_session")
            final_message = Message(
                role=RoleEnum.USER,
                content=final_prompt,
                message_type=MessageType.TEXT
            )
            final_history.add_message(final_message)

            response = await llm_provider.generate_response(history=final_history)
            llm_output = response.content.strip()
        
    else:
        # Content is small enough to process directly
        prompt = summary_prompts[summary_type].format(content=content)
        
        history = History(session_id="summarization_session")
        user_message = Message(
            role=RoleEnum.USER,
            content=prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)

        if use_structured_output:
            response = await llm_provider.generate_structured_response(
                history=history,
                response_format=TranscriptionSummary
            )
            # Extract structured content from the response
            if hasattr(response, '_structured_content'):
                structured_result = response._structured_content
            else:
                # Fallback: parse from JSON content
                import json
                content_dict = json.loads(response.content)
                structured_result = TranscriptionSummary(**content_dict)
        else:
            response = await llm_provider.generate_response(history=history)
            llm_output = response.content.strip()

    # Debug info: print LLM response stats
    if use_structured_output:
        print(f"Structured output received:")
        print(f"Summary length: {len(structured_result.summary)} characters")
        print(f"Key points: {len(structured_result.key_points)}")
        print(f"Actions: {len(structured_result.actions)}")
        print(f"Language detected: {structured_result.language_detected}")
    else:
        print(f"LLM response length: {len(llm_output)} characters")
        print(f"LLM response first 300 characters: {llm_output[:300]}...")

    if summary_type == "comprehensive":
        if use_structured_output:
            # Use structured output directly - no parsing needed!
            summary = structured_result.summary
            key_points = structured_result.key_points if extract_key_points else []
            actions = structured_result.actions if extract_actions else []
            
            # Filter out empty actions or "no actions" statements
            actions = [action for action in actions if action and not action.lower().strip().startswith("no specific actions")]
        else:
            # Fallback to old parsing method for compatibility
            summary = ""
            key_points = []
            actions = []
            
            sections = llm_output.split("\n\n")
            current_section = ""
            
            for section in sections:
                section = section.strip()
                # Handle both English and Armenian section headers
                if (section.startswith("SUMMARY:") or section.startswith("**ԱՄՓՈՓՈՒՄ:**") or 
                    section.startswith("ԱՄՓՈՓՈՒՄ:") or section.startswith("**SUMMARY:**")):
                    current_section = "summary"
                    # Remove various header formats
                    summary = section
                    for header in ["SUMMARY:", "**ԱՄՓՈՓՈՒՄ:**", "ԱՄՓՈՓՈՒՄ:", "**SUMMARY:**"]:
                        summary = summary.replace(header, "").strip()
                elif (section.startswith("KEY POINTS:") or section.startswith("**ՀԻՄՆԱԿԱՆ ԿԵՏԵՐ:**") or 
                        section.startswith("ՀԻՄՆԱԿԱՆ ԿԵՏԵՐ:") or section.startswith("**KEY POINTS:**")):
                    current_section = "key_points"
                elif (section.startswith("ACTIONS:") or section.startswith("**ԳՈՐԾՈՂՈՒԹՅՈՒՆՆԵՐ:**") or 
                        section.startswith("ԳՈՐԾՈՂՈՒԹՅՈՒՆՆԵՐ:") or section.startswith("**ACTIONS:**")):
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

        # Debug info: print parsing results
        print(f"Final summary length: {len(summary)} characters")
        print(f"Key points found: {len(key_points)}")
        print(f"Actions found: {len(actions)}")
        print(f"Summary content: {summary[:200]}...")

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
                "file_size_bytes": file_path.stat().st_size,
                "language_detected": structured_result.language_detected if use_structured_output else "unknown",
                "used_structured_output": use_structured_output
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


@log_tool_call("get_transcription_summaries")
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


@log_tool_call("translate_transcription_file")
async def translate_transcription_file_tool(
    filename: str,
    target_language: str = "english",
    source_language: str = "auto-detect"
) -> Dict[str, Any]:
    """Tool for translating transcription files to target language using LLM."""
    transcription_paths = [
        Path("../data/transcriptions"),  # From api directory
        Path("data/transcriptions"),      # From project root
        Path("/Users/sadamyan/workdir/agentic-asr/data/transcriptions"),  # Absolute path
    ]
    
    file_path = None
    for base_path in transcription_paths:
        # First try exact match
        potential_path = base_path / filename
        if potential_path.exists():
            file_path = potential_path
            break
        
        # Try with different extensions if not already has one
        for ext in ['.txt', '.json', '.md']:
            if not filename.endswith(ext):
                potential_path = base_path / f"{filename}{ext}"
                if potential_path.exists():
                    file_path = potential_path
                    break
        if file_path:
            break
            
        # Try fuzzy matching for common variations (spaces, case, etc.)
        if not file_path and base_path.exists():
            import re
            # Create a pattern that allows for flexible matching
            normalized_filename = re.sub(r'[^\w]', '', filename.lower())
            for file_candidate in base_path.glob('*'):
                if file_candidate.is_file():
                    candidate_normalized = re.sub(r'[^\w]', '', file_candidate.stem.lower())
                    if normalized_filename in candidate_normalized or candidate_normalized in normalized_filename:
                        # Additional check for common patterns like "Rearrange #261" vs "Rearrange#261"
                        if abs(len(normalized_filename) - len(candidate_normalized)) <= 2:
                            file_path = file_candidate
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

    # Check if content needs chunking for translation
    estimated_tokens = len(content) // 4  # Rough estimate: 1 token ≈ 4 characters

    llm_config = LLMProviderConfig(
        provider_name="openai",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )
    llm_provider = create_llm_provider(llm_config)

    if estimated_tokens > 100000:  # If content is too large, chunk it
        chunks = chunk_text(content, max_tokens=25000)  # Smaller chunks for translation
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            if source_language == "auto-detect":
                chunk_prompt = f"""Please translate the following transcription text to {target_language}. 
If the source text is already in {target_language}, please respond with the original text.
Maintain the original meaning, tone, and structure as much as possible.

Text to translate:
{chunk}

Please provide only the translated text without explanations."""
            else:
                chunk_prompt = f"""Please translate the following transcription text from {source_language} to {target_language}.
Maintain the original meaning, tone, and structure as much as possible.

Text to translate:
{chunk}

Please provide only the translated text without explanations."""
            
            history = History(session_id=f"translation_session_chunk_{i}")
            user_message = Message(
                role=RoleEnum.USER,
                content=chunk_prompt,
                message_type=MessageType.TEXT
            )
            history.add_message(user_message)

            try:
                response = await llm_provider.generate_response(history=history)
                translated_chunks.append(response.content.strip())
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to translate chunk {i+1}/{len(chunks)}: {str(e)}"
                }
        
        # Combine translated chunks with appropriate joining
        translated_text = " ".join(translated_chunks)  # Use space for better flow in translations
        
    else:
        # Content is small enough to process directly
        if source_language == "auto-detect":
            translation_prompt = f"""Please translate the following transcription text to {target_language}. 
If the source text is already in {target_language}, please respond with the original text.
Maintain the original meaning, tone, and structure as much as possible.

Text to translate:
{content}

Please provide only the translated text without explanations."""
        else:
            translation_prompt = f"""Please translate the following transcription text from {source_language} to {target_language}.
Maintain the original meaning, tone, and structure as much as possible.

Text to translate:
{content}

Please provide only the translated text without explanations."""
        
        history = History(session_id="translation_session")
        user_message = Message(
            role=RoleEnum.USER,
            content=translation_prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)

        response = await llm_provider.generate_response(history=history)
        translated_text = response.content.strip()

    original_length = len(content)
    translated_length = len(translated_text)
    
    original_words = content.split()
    translated_words = translated_text.split()

    result = {
        "success": True,
        "file_path": str(file_path),
        "filename": filename,
        "source_language": source_language,
        "target_language": target_language,
        "original_text": content,
        "translated_text": translated_text,
        "metadata": {
            "original_length": original_length,
            "translated_length": translated_length,
            "original_word_count": len(original_words),
            "translated_word_count": len(translated_words),
            "file_size_bytes": file_path.stat().st_size,
            "translation_ratio": translated_length / original_length if original_length > 0 else 0
        }
    }
    print("result", result)

    try:
        history_manager = HistoryManager()
        translation_id = await history_manager.save_transcription_translation(
            filename=filename,
            file_path=str(file_path),
            source_language=source_language,
            target_language=target_language,
            original_text=content,
            translated_text=translated_text,
            metadata=result["metadata"]
        )
        result["database_id"] = translation_id
        result["saved_to_database"] = True
        await history_manager.close()
    except Exception as db_error:
        result["database_error"] = str(db_error)
        result["saved_to_database"] = False

    return result

@log_tool_call("get_transcription_translations")
async def get_transcription_translations_tool(
    filename: str = None,
    target_language: str = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Tool to retrieve saved transcription translations from database."""
    try:
        history_manager = HistoryManager()
        translations = await history_manager.get_transcription_translations(
            filename=filename, 
            target_language=target_language, 
            limit=limit
        )
        await history_manager.close()
        
        return {
            "success": True,
            "translations": translations,
            "count": len(translations),
            "filename_filter": filename,
            "target_language_filter": target_language
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to retrieve translations: {str(e)}"
        }


@log_tool_call("get_transcription_content")
async def get_transcription_content_tool(
    filename: str,
    include_metadata: bool = True,
    max_content_length: int = 100000
) -> Dict[str, Any]:
    """Tool to retrieve the content of a transcription file (supports only .txt files)."""
    transcription_paths = [
        Path("../data/transcriptions"),  # From api directory
        Path("data/transcriptions"),      # From project root
        Path("/Users/sadamyan/workdir/agentic-asr/data/transcriptions"),  # Absolute path
    ]
    
    file_path = None
    for base_path in transcription_paths:
        # First try exact match with .txt extension
        if filename.endswith('.txt'):
            potential_path = base_path / filename
        else:
            potential_path = base_path / f"{filename}.txt"
            
        if potential_path.exists():
            file_path = potential_path
            break
            
        # Try fuzzy matching for common variations (spaces, case, etc.)
        if not file_path and base_path.exists():
            # Create a pattern that allows for flexible matching
            normalized_filename = re.sub(r'[^\w]', '', filename.lower())
            for file_candidate in base_path.glob('*.txt'):
                if file_candidate.is_file():
                    candidate_normalized = re.sub(r'[^\w]', '', file_candidate.stem.lower())
                    if normalized_filename in candidate_normalized or candidate_normalized in normalized_filename:
                        # Additional check for common patterns like "Rearrange #261" vs "Rearrange#261"
                        if abs(len(normalized_filename) - len(candidate_normalized)) <= 3:
                            file_path = file_candidate
                            break
        if file_path:
            break
    
    if not file_path:
        return {
            "success": False,
            "error": f"Transcription file '{filename}' not found in any of the transcription directories: {[str(p) for p in transcription_paths]}. Only .txt files are supported."
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

    # Check if content is too large and needs to be truncated or chunked
    estimated_tokens = len(content) // 4  # Rough estimate: 1 token ≈ 4 characters
    is_large_content = estimated_tokens > 25000  # Conservative limit to prevent context overflow
    
    # For very large content, provide options
    content_to_return = content
    content_warning = None
    chunks_info = None
    
    if is_large_content:
        if len(content) > max_content_length:
            # Truncate content but provide information about the full content
            content_to_return = content[:max_content_length] + "\n\n[CONTENT TRUNCATED - File is too large to display in full]"
            content_warning = f"Content truncated to {max_content_length} characters. Original content was {len(content)} characters."
        
        # Also provide chunking information
        chunks = chunk_text(content, max_tokens=20000)  # Smaller chunks for safer processing
        chunks_info = {
            "total_chunks": len(chunks),
            "chunk_sizes": [len(chunk) for chunk in chunks],
            "chunks_available": True,
            "recommendation": "Consider processing this file in chunks using other tools for better results."
        }

    result = {
        "success": True,
        "filename": filename,
        "file_path": str(file_path),
        "content": content_to_return,
        "is_large_content": is_large_content,
        "estimated_tokens": estimated_tokens
    }
    
    if content_warning:
        result["content_warning"] = content_warning
    
    if chunks_info:
        result["chunks_info"] = chunks_info

    if include_metadata:
        from datetime import datetime
        
        file_stat = file_path.stat()
        
        # Calculate basic text statistics (use original content for accurate stats)
        lines = content.count('\n') + 1 if content else 0
        words = len(content.split())
        characters = len(content)
        
        # Detect potential language (basic heuristic) - sample from beginning if truncated
        sample_text = content[:10000] if len(content) > 10000 else content
        detected_language = "unknown"
        armenian_chars = sum(1 for char in sample_text if '\u0530' <= char <= '\u058F')
        latin_chars = sum(1 for char in sample_text if char.isalpha() and ord(char) < 256)
        
        if armenian_chars > latin_chars:
            detected_language = "armenian"
        elif latin_chars > armenian_chars:
            detected_language = "english/latin"
        
        result["metadata"] = {
            "file_size_bytes": file_stat.st_size,
            "created_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "character_count": characters,
            "word_count": words,
            "line_count": lines,
            "estimated_reading_time_minutes": max(1, words // 200),  # Average reading speed ~200 wpm
            "detected_language": detected_language,
            "armenian_characters": armenian_chars,
            "latin_characters": latin_chars,
            "content_truncated": len(content_to_return) < len(content)
        }

    return result


# RAG and Vector Store Tools

@log_tool_call("process_transcription_for_rag")
async def process_transcription_for_rag_tool(
    filename: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Process a transcription file for RAG by chunking and embedding it in the local vector store."""
    from ..vector_store import get_vector_store
    from pathlib import Path
    import re
    
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Find the transcription file using the same robust logic as other tools
        transcription_paths = [
            Path("../data/transcriptions"),  # From api directory
            Path("data/transcriptions"),      # From project root
            Path("/Users/sadamyan/workdir/agentic-asr/data/transcriptions"),  # Absolute path
        ]
        
        file_path = None
        for base_path in transcription_paths:
            # First try exact match
            potential_path = base_path / filename
            if potential_path.exists():
                file_path = potential_path
                break
            
            # Try with different extensions if not already has one
            for ext in ['.txt', '.json', '.md']:
                if not filename.endswith(ext):
                    potential_path = base_path / f"{filename}{ext}"
                    if potential_path.exists():
                        file_path = potential_path
                        break
            if file_path:
                break
                
            # Try fuzzy matching for common variations (spaces, case, etc.)
            if not file_path and base_path.exists():
                # Create a pattern that allows for flexible matching
                normalized_filename = re.sub(r'[^\w]', '', filename.lower())
                for file_candidate in base_path.glob('*'):
                    if file_candidate.is_file():
                        candidate_normalized = re.sub(r'[^\w]', '', file_candidate.stem.lower())
                        if normalized_filename in candidate_normalized or candidate_normalized in normalized_filename:
                            # Additional check for common patterns like "Rearrange #261" vs "Rearrange#261"
                            if abs(len(normalized_filename) - len(candidate_normalized)) <= 2:
                                file_path = file_candidate
                                break
            if file_path:
                break
        
        if not file_path:
            return {
                "success": False,
                "error": f"Transcription file '{filename}' not found in any of the transcription directories: {[str(p) for p in transcription_paths]}"
            }
        
        # Read the transcription content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return {
                "success": False,
                "error": f"Transcription file '{filename}' is empty"
            }
        
        # Use provided chunk parameters or defaults
        chunk_size = chunk_size or 1000
        overlap = overlap or 200
        
        # Process the document
        result = vector_store.add_document(
            filename=filename,
            content=content,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        return result
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing transcription for RAG: {str(e)}"
        }


@log_tool_call("semantic_search_transcriptions")
async def semantic_search_transcriptions_tool(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.1,
    include_content: bool = True
) -> Dict[str, Any]:
    """Search transcriptions using semantic similarity via local vector store."""
    from ..vector_store import get_vector_store
    
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Perform the search
        results = vector_store.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results for better readability
        formatted_results = []
        for search_result in results:
            formatted_result = {
                "filename": search_result.get("filename"),
                "chunk_index": search_result.get("chunk_index"),
                "similarity_score": round(search_result.get("similarity_score", 0), 4),
                "content": search_result.get("content") if include_content else None,
                "document_id": search_result.get("document_id"),
                "chunk_id": search_result.get("chunk_id"),
                "metadata": {}
            }
            formatted_results.append(formatted_result)
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_time_ms": 0,  # Not tracking time in simple implementation
            "index_used": "default"
        }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Error performing semantic search: {str(e)}"
        }


@log_tool_call("get_relevant_context")
async def get_relevant_context_tool(
    query: str,
    max_chunks: int = 3,
    similarity_threshold: float = 0.2
) -> Dict[str, Any]:
    """Get relevant context chunks for a query, formatted for use in LLM prompts."""
    try:
        # Search for relevant chunks
        search_result = await semantic_search_transcriptions_tool(
            query=query,
            top_k=max_chunks,
            similarity_threshold=similarity_threshold,
            include_content=True
        )
        
        if not search_result.get("success"):
            return search_result
        
        results = search_result.get("results", [])
        
        if not results:
            return {
                "success": True,
                "query": query,
                "context": "",
                "context_chunks": [],
                "total_chunks": 0,
                "message": "No relevant context found for the query."
            }
        
        # Format context for LLM use
        context_parts = []
        context_chunks = []
        
        for i, result in enumerate(results, 1):
            filename = result.get("filename", "Unknown")
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0)
            
            if content:
                # Clean filename for display
                clean_filename = filename.replace(".txt", "").replace("Rearrange #", "Episode ")
                
                context_part = f"[Context {i} from {clean_filename} (similarity: {similarity:.3f})]:\n{content.strip()}\n"
                context_parts.append(context_part)
                
                context_chunks.append({
                    "filename": filename,
                    "content": content,
                    "similarity_score": similarity,
                    "chunk_index": result.get("chunk_index"),
                    "document_id": result.get("document_id")
                })
        
        formatted_context = "\n".join(context_parts)
        
        return {
            "success": True,
            "query": query,
            "context": formatted_context,
            "context_chunks": context_chunks,
            "total_chunks": len(context_chunks),
            "search_time_ms": search_result.get("search_time_ms"),
            "usage_instruction": "Use this context to answer questions about the transcribed content. The context is sorted by relevance."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting relevant context: {str(e)}"
        }


@log_tool_call("rag_answer_question")
async def rag_answer_question_tool(
    question: str,
    max_context_chunks: int = 3,
    similarity_threshold: float = 0.2
) -> Dict[str, Any]:
    """Answer a question using RAG (Retrieval-Augmented Generation) with transcription data."""
    try:
        # Get relevant context
        context_result = await get_relevant_context_tool(
            query=question,
            max_chunks=max_context_chunks,
            similarity_threshold=similarity_threshold
        )
        
        if not context_result.get("success"):
            return context_result
        
        context = context_result.get("context", "")
        context_chunks = context_result.get("context_chunks", [])
        
        if not context.strip():
            return {
                "success": True,
                "question": question,
                "answer": "I couldn't find relevant information in the transcriptions to answer your question.",
                "context_used": [],
                "confidence": "low",
                "sources": []
            }
        
        # Use LLM to generate answer based on context
        llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )
        
        llm_provider = create_llm_provider(llm_config)
        
        # Create RAG prompt
        rag_prompt = f"""You are an AI assistant that answers questions based on transcription data from podcasts and interviews. 

Use the following context from transcriptions to answer the user's question. The context is sorted by relevance to the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question based ONLY on the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite which sources/episodes your answer comes from when possible
4. Maintain the same language as the context (if context is in Armenian, answer in Armenian, etc.)
5. Be specific and provide details when available
6. If multiple sources mention different aspects, synthesize them coherently

ANSWER:"""
        
        history = History(session_id="rag_session")
        user_message = Message(
            role=RoleEnum.USER,
            content=rag_prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)
        
        response = await llm_provider.generate_response(history=history)
        answer = response.content.strip()
        
        # Determine confidence based on context quality
        avg_similarity = sum(chunk.get("similarity_score", 0) for chunk in context_chunks) / len(context_chunks) if context_chunks else 0
        
        if avg_similarity > 0.4:
            confidence = "high"
        elif avg_similarity > 0.25:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Extract sources
        sources = []
        for chunk in context_chunks:
            filename = chunk.get("filename", "Unknown")
            clean_filename = filename.replace(".txt", "").replace("Rearrange #", "Episode ")
            sources.append({
                "filename": clean_filename,
                "similarity": chunk.get("similarity_score"),
                "chunk_index": chunk.get("chunk_index")
            })
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "context_used": context_chunks,
            "confidence": confidence,
            "sources": sources,
            "total_context_chunks": len(context_chunks),
            "average_similarity": round(avg_similarity, 3),
            "search_time_ms": context_result.get("search_time_ms")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating RAG answer: {str(e)}"
        }


@log_tool_call("process_all_transcriptions_for_rag")
async def process_all_transcriptions_for_rag_tool(
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Process all transcription files for RAG."""
    from ..vector_store import get_vector_store
    from pathlib import Path
    
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Find all transcription files using the same robust path logic
        transcription_paths = [
            Path("../data/transcriptions"),  # From api directory
            Path("data/transcriptions"),      # From project root
            Path("/Users/sadamyan/workdir/agentic-asr/data/transcriptions"),  # Absolute path
        ]
        
        transcription_files = []
        for base_path in transcription_paths:
            if base_path.exists():
                # Look for text files with various extensions
                for ext in ['*.txt', '*.json', '*.md']:
                    transcription_files.extend(base_path.glob(ext))
        
        if not transcription_files:
            return {
                "success": False,
                "error": f"No transcription files found in any of the transcription directories: {[str(p) for p in transcription_paths]}"
            }
        
        # Use provided chunk parameters or defaults
        chunk_size = chunk_size or 1000
        overlap = overlap or 200
        
        processed_count = 0
        errors = []
        
        for file_path in transcription_files:
            try:
                filename = file_path.name
                
                # Read the transcription content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    # Process the document
                    result = vector_store.add_document(
                        filename=filename,
                        content=content,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    
                    if result.get("success"):
                        processed_count += 1
                    else:
                        errors.append(f"{filename}: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
        
        return {
            "success": True,
            "message": f"Processed {processed_count} transcription files",
            "files_count": processed_count,
            "total_files_found": len(transcription_files),
            "errors": errors,
            "status": "completed"
        }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing transcriptions: {str(e)}"
        }


@log_tool_call("get_vector_store_stats")
async def get_vector_store_stats_tool() -> Dict[str, Any]:
    """Get statistics about the vector store."""
    from ..vector_store import get_vector_store
    
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Get statistics
        stats = vector_store.get_stats()
        documents = vector_store.list_documents()
        
        return {
            "success": True,
            "stats": {
                "total_vectors": stats["total_vectors"],
                "total_documents": stats["total_documents"],
                "total_chunks": stats["total_chunks"],
                "model_name": stats["model_name"],
                "store_path": stats["store_path"],
                "index_file_size_bytes": stats["index_file_size"]
            },
            "documents": documents
        }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting vector store stats: {str(e)}"
        }


def get_enhanced_tools() -> list[ToolDefinition]:
    """Get enhanced tools including real transcription capabilities and RAG functionality."""
    return [
        ToolDefinition(
            name="analyze_transcription",
            description="Perform advanced analysis on transcribed text including summarization, keyword extraction, and sentiment analysis",
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
            description="Correct transcription errors using LLM intelligence with multiple correction levels and context awareness",
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
                        "description": "Level of correction to apply: 'light' (basic errors), 'medium' (grammar and readability), or 'heavy' (comprehensive correction)",
                        "enum": ["light", "medium", "heavy"],
                        "default": "medium"
                    }
                },
                "required": ["original_text"]
            },
            function=correct_transcription_tool
        ),
        ToolDefinition(
            name="summarize_transcription_file",
            description="Summarize transcription files with intelligent language detection, key point extraction, and action item identification. Handles large files through chunking.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to summarize (searches in data/transcriptions directories)"
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summary: 'brief' (2-3 sentences), 'detailed' (comprehensive overview), or 'comprehensive' (includes key points and actions)",
                        "enum": ["brief", "detailed", "comprehensive"],
                        "default": "comprehensive"
                    },
                    "extract_actions": {
                        "type": "boolean",
                        "description": "Whether to extract action items and tasks mentioned in the transcription",
                        "default": True
                    },
                    "extract_key_points": {
                        "type": "boolean",
                        "description": "Whether to extract key discussion points and important topics",
                        "default": True
                    },
                    "max_summary_length": {
                        "type": "integer",
                        "description": "Maximum desired length of the summary in characters",
                        "default": 500
                    }
                },
                "required": ["filename"]
            },
            function=summarize_transcription_file_tool
        ),
        ToolDefinition(
            name="get_transcription_summaries",
            description="Retrieve previously saved transcription summaries from the database with optional filtering",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter to get summaries for a specific transcription file only"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of summaries to retrieve (default: 10)",
                        "default": 10
                    }
                },
                "required": []
            },
            function=get_transcription_summaries_tool
        ),
        ToolDefinition(
            name="translate_transcription_file",
            description="Translate transcription files to target languages using LLM with intelligent language detection and chunking for large files",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to translate (searches in data/transcriptions directories)"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language for translation (e.g., 'english', 'spanish', 'french', 'german', 'armenian', 'russian', etc.)",
                        "default": "english"
                    },
                    "source_language": {
                        "type": "string",
                        "description": "Source language of the transcription (use 'auto-detect' to let LLM automatically detect the language)",
                        "default": "auto-detect"
                    }
                },
                "required": ["filename"]
            },
            function=translate_transcription_file_tool
        ),
        ToolDefinition(
            name="get_transcription_translations",
            description="Retrieve previously saved transcription translations from the database with filtering options",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter to get translations for a specific transcription file"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Optional target language filter to get translations in a specific language only"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of translations to retrieve (default: 10)",
                        "default": 10
                    }
                },
                "required": []
            },
            function=get_transcription_translations_tool
        ),
        ToolDefinition(
            name="get_transcription_content",
            description="Retrieve the raw content of a transcription file with metadata and handling for large files",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to translate (searches in data/transcriptions directories)"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Whether to include file metadata (size, dates, word count, language detection)",
                        "default": True
                    },
                    "max_content_length": {
                        "type": "integer",
                        "description": "Maximum content length to return (larger files will be truncated with warning)",
                        "default": 100000
                    }
                },
                "required": ["filename"]
            },
            function=get_transcription_content_tool
        ),
        ToolDefinition(
            name="process_transcription_for_rag",
            description="Process a transcription file for RAG (Retrieval-Augmented Generation) by chunking and embedding it in the vector store",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to translate (searches in data/transcriptions directories)"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks for processing (default: 1000 characters)"
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks to maintain context (default: 200 characters)"
                    }
                },
                "required": ["filename"]
            },
            function=process_transcription_for_rag_tool
        ),
        ToolDefinition(
            name="semantic_search_transcriptions",
            description="Search across all processed transcriptions using semantic similarity to find relevant content",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find semantically similar content in transcriptions"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of most similar results to return (default: 5)",
                        "default": 5
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0.0 to 1.0, default: 0.1)",
                        "default": 0.1
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to include the actual content in results",
                        "default": True
                    }
                },
                "required": ["query"]
            },
            function=semantic_search_transcriptions_tool
        ),
        ToolDefinition(
            name="get_relevant_context",
            description="Get relevant context chunks for a query, formatted for use in LLM prompts and conversations",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to find relevant context for"
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of context chunks to retrieve (default: 3)",
                        "default": 3
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score for context relevance (default: 0.2)",
                        "default": 0.2
                    }
                },
                "required": ["query"]
            },
            function=get_relevant_context_tool
        ),
        ToolDefinition(
            name="rag_answer_question",
            description="Answer questions using RAG (Retrieval-Augmented Generation) by finding relevant transcription content and generating informed responses",
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer using transcription data"
                    },
                    "max_context_chunks": {
                        "type": "integer",
                        "description": "Maximum number of context chunks to use for answering (default: 3)",
                        "default": 3
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for relevant context (default: 0.2)",
                        "default": 0.2
                    }
                },
                "required": ["question"]
            },
            function=rag_answer_question_tool
        ),
        ToolDefinition(
            name="process_all_transcriptions_for_rag",
            description="Process all available transcription files for RAG by chunking and embedding them in the vector store",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks for processing (default: 1000 characters)"
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks to maintain context (default: 200 characters)"
                    }
                },
                "required": []
            },
            function=process_all_transcriptions_for_rag_tool
        ),
        ToolDefinition(
            name="get_vector_store_stats",
            description="Get statistics and information about the current vector store including document count, vector count, and storage details",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=get_vector_store_stats_tool
        )
    ]
