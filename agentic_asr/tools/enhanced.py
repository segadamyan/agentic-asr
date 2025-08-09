"""Integration with the transcriber to create a real transcription tool."""
import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel
import json
import os
import re

from ..core.history import HistoryManager
from ..core.models import ToolDefinition, LLMProviderConfig, History, Message, RoleEnum, MessageType
from ..llm.providers import create_llm_provider


# Pydantic models for structured outputs
class TranscriptionSummary(BaseModel):
    """Structured output model for transcription summarization."""
    summary: str
    key_points: List[str]
    actions: List[str]
    language_detected: str


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
        ),
        ToolDefinition(
            name="translate_transcription_file",
            description="Translate transcription files to target language using LLM",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the transcription file to translate (will search in transcription directories)"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language for translation (e.g., 'english', 'spanish', 'french', 'german', 'armenian', etc.)",
                        "default": "english"
                    },
                    "source_language": {
                        "type": "string",
                        "description": "Source language of the transcription (use 'auto-detect' to let LLM detect)",
                        "default": "auto-detect"
                    }
                },
                "required": ["filename"]
            },
            function=translate_transcription_file_tool
        ),
        ToolDefinition(
            name="get_transcription_translations",
            description="Retrieve saved transcription translations from database",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter to get translations for a specific file"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Optional target language filter to get translations for a specific language"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of translations to retrieve",
                        "default": 10
                    }
                },
                "required": []
            },
            function=get_transcription_translations_tool
        )
    ]
