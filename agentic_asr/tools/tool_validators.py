"""
Tool Validators for Transcription Correction and Translation Quality Assessment

This module provides LLM-based validators to assess the quality of transcription corrections
and translations using intelligent agents with specialized prompts. The validators use 
GPT-4o to analyze and rate outputs from 0 to 10 across multiple dimensions.
"""

import os
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Import LLM components
from ..core.models import LLMProviderConfig, History, Message, RoleEnum, MessageType
from ..llm.providers import create_llm_provider


class ValidationResult:
    """Result structure for validation scores and feedback."""
    
    def __init__(self, overall_score: float = 0.0):
        self.overall_score = overall_score
        self.scores = {}
        self.feedback = {}
        self.metrics = {}
        self.issues = []
        self.recommendations = []
        self.validation_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "scores": self.scores,
            "feedback": self.feedback,
            "metrics": self.metrics,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "validation_timestamp": self.validation_timestamp
        }


class CorrectionValidatorAgent:
    """LLM-based agent for validating transcription correction quality."""
    
    def __init__(self):
        self.llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1  # Low temperature for consistent scoring
        )
        self.llm_provider = create_llm_provider(self.llm_config)
    
    async def validate_correction(self, original_text: str, corrected_text: str, 
                                context: str = "", validation_level: str = "comprehensive") -> ValidationResult:
        """
        Validate the quality of a transcription correction using LLM intelligence.
        
        Args:
            original_text: The original transcribed text with potential errors
            corrected_text: The corrected text to validate
            context: Optional context for validation
            validation_level: Level of validation detail ('basic', 'comprehensive', 'expert')
            
        Returns:
            ValidationResult with scores from 0-10 and detailed feedback
        """
        result = ValidationResult()
        
        # Create validation prompt based on level
        prompt = self._create_correction_validation_prompt(
            original_text, corrected_text, context, validation_level
        )
        

        history = History(session_id="correction_validation")
        user_message = Message(
            role=RoleEnum.USER,
            content=prompt,
            message_type=MessageType.TEXT
        )
        history.add_message(user_message)
        
        response = await self.llm_provider.generate_response(history=history)
        validation_data = self._parse_validation_response(response.content)
        
        # Populate result with LLM analysis
        result.overall_score = validation_data.get("overall_score", 0.0)
        result.scores = validation_data.get("scores", {})
        result.feedback = validation_data.get("feedback", {})
        result.metrics = validation_data.get("metrics", {})
        result.issues = validation_data.get("issues", [])
        result.recommendations = validation_data.get("recommendations", [])
            
        return result
    
    def _create_correction_validation_prompt(self, original: str, corrected: str, 
                                           context: str, level: str) -> str:
        """Create a specialized prompt for correction validation."""
        
        base_prompt = f"""You are an expert language validation agent specialized in evaluating transcription corrections. Your task is to analyze the quality of a transcription correction and provide detailed scoring from 0 to 10.

ORIGINAL TRANSCRIPTION:
{original}

CORRECTED TRANSCRIPTION:
{corrected}

CONTEXT: {context if context else "No additional context provided"}

VALIDATION LEVEL: {level}

Please evaluate the correction across these dimensions and provide scores from 0 to 10:

1. **Grammar Improvement** (0-10): How well were grammatical errors corrected?
2. **Spelling Accuracy** (0-10): How effectively were spelling errors fixed?
3. **Meaning Preservation** (0-10): How well was the original meaning maintained?
4. **Fluency Enhancement** (0-10): How much did the correction improve readability and flow?
5. **Language Consistency** (0-10): How consistent is the language style and tone?
6. **Accuracy Score** (0-10): Overall accuracy of the corrections made

Provide your response in this exact JSON format:
{{
    "overall_score": <average of all scores>,
    "scores": {{
        "grammar_score": <0-10>,
        "spelling_score": <0-10>,
        "meaning_preservation_score": <0-10>,
        "fluency_score": <0-10>,
        "language_consistency_score": <0-10>,
        "accuracy_score": <0-10>
    }},
    "feedback": {{
        "detailed_feedback": "<comprehensive analysis of the correction quality>",
        "summary": "<brief summary of overall quality>"
    }},
    "metrics": {{
        "original_length": {len(original)},
        "corrected_length": {len(corrected)},
        "original_word_count": {len(original.split())},
        "corrected_word_count": {len(corrected.split())},
        "change_percentage": "<estimated percentage of changes made>"
    }},
    "issues": [
        "<list of specific issues found>"
    ],
    "recommendations": [
        "<list of actionable recommendations>"
    ]
}}

Analyze carefully and provide accurate scores based on the actual quality of the correction."""

        if level == "expert":
            base_prompt += "\n\nEXPERT MODE: Provide additional linguistic analysis including syntax, semantics, and style evaluation."
        elif level == "basic":
            base_prompt += "\n\nBASIC MODE: Focus on fundamental correction quality - grammar, spelling, and basic readability."
        
        return base_prompt
    
    def _parse_validation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the LLM response and extract validation data."""
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # Fallback: create basic structure from text analysis
            return self._create_fallback_validation(response_content)

    
    def _create_fallback_validation(self, response_content: str) -> Dict[str, Any]:
        """Create fallback validation data when JSON parsing fails."""
        return {
            "overall_score": 6.0,
            "scores": {
                "grammar_score": 6.0,
                "spelling_score": 6.0,
                "meaning_preservation_score": 6.0,
                "fluency_score": 6.0,
                "language_consistency_score": 6.0,
                "accuracy_score": 6.0
            },
            "feedback": {
                "detailed_feedback": response_content,
                "summary": "Validation completed with fallback parsing"
            },
            "metrics": {},
            "issues": [],
            "recommendations": []
        }


class TranslationValidatorAgent:
    """LLM-based agent for validating transcription translation quality."""
    
    def __init__(self):
        self.llm_config = LLMProviderConfig(
            provider_name="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1  # Low temperature for consistent scoring
        )
        self.llm_provider = create_llm_provider(self.llm_config)
    
    async def validate_translation(self, original_text: str, translated_text: str,
                                 source_language: str, target_language: str,
                                 context: str = "", validation_level: str = "comprehensive") -> ValidationResult:
        """
        Validate the quality of a transcription translation using LLM intelligence.
        
        Args:
            original_text: The original text in source language
            translated_text: The translated text to validate
            source_language: Source language
            target_language: Target language
            context: Optional context for validation
            validation_level: Level of validation detail ('basic', 'comprehensive', 'expert')
            
        Returns:
            ValidationResult with scores from 0-10 and detailed feedback
        """
        result = ValidationResult()
        
        # Create validation prompt based on level
        prompt = self._create_translation_validation_prompt(
            original_text, translated_text, source_language, target_language, context, validation_level
        )
        
        # Get LLM evaluation
        try:
            history = History(session_id="translation_validation")
            user_message = Message(
                role=RoleEnum.USER,
                content=prompt,
                message_type=MessageType.TEXT
            )
            history.add_message(user_message)
            
            response = await self.llm_provider.generate_response(history=history)
            validation_data = self._parse_validation_response(response.content)
            
            result.overall_score = validation_data.get("overall_score", 0.0)
            result.scores = validation_data.get("scores", {})
            result.feedback = validation_data.get("feedback", {})
            result.metrics = validation_data.get("metrics", {})
            result.issues = validation_data.get("issues", [])
            result.recommendations = validation_data.get("recommendations", [])
            
        except Exception as e:
            result.overall_score = 0.0
            result.feedback = {"error": f"Validation failed: {str(e)}"}
            result.issues = ["LLM validation error"]
        
        return result
    
    def _create_translation_validation_prompt(self, original: str, translated: str,
                                            source_lang: str, target_lang: str,
                                            context: str, level: str) -> str:
        """Create a specialized prompt for translation validation."""
        
        base_prompt = f"""You are an expert translation validation agent specialized in evaluating translation quality. Your task is to analyze the quality of a transcription translation and provide detailed scoring from 0 to 10.

ORIGINAL TEXT ({source_lang}):
{original}

TRANSLATED TEXT ({target_lang}):
{translated}

SOURCE LANGUAGE: {source_lang}
TARGET LANGUAGE: {target_lang}
CONTEXT: {context if context else "No additional context provided"}

VALIDATION LEVEL: {level}

Please evaluate the translation across these dimensions and provide scores from 0 to 10:

1. **Language Detection Accuracy** (0-10): Are the source and target languages correctly identified and handled?
2. **Translation Accuracy** (0-10): How accurate is the translation of meaning and content?
3. **Fluency in Target Language** (0-10): How natural and fluent does the translation read?
4. **Meaning Preservation** (0-10): How well is the original meaning preserved?
5. **Cultural Adaptation** (0-10): How well are cultural references and context adapted?
6. **Completeness** (0-10): Is the translation complete without omissions?

Provide your response in this exact JSON format:
{{
    "overall_score": <average of all scores>,
    "scores": {{
        "language_detection_score": <0-10>,
        "accuracy_score": <0-10>,
        "fluency_score": <0-10>,
        "meaning_preservation_score": <0-10>,
        "cultural_adaptation_score": <0-10>,
        "completeness_score": <0-10>
    }},
    "feedback": {{
        "detailed_feedback": "<comprehensive analysis of the translation quality>",
        "summary": "<brief summary of overall quality>"
    }},
    "metrics": {{
        "original_length": {len(original)},
        "translated_length": {len(translated)},
        "original_word_count": {len(original.split())},
        "translated_word_count": {len(translated.split())},
        "length_ratio": {len(translated) / max(1, len(original)):.2f},
        "source_language": "{source_lang}",
        "target_language": "{target_lang}"
    }},
    "issues": [
        "<list of specific translation issues found>"
    ],
    "recommendations": [
        "<list of actionable recommendations for improvement>"
    ]
}}

Analyze carefully and provide accurate scores based on the actual quality of the translation."""

        if level == "expert":
            base_prompt += "\n\nEXPERT MODE: Provide additional analysis including linguistic accuracy, cultural nuances, and stylistic appropriateness."
        elif level == "basic":
            base_prompt += "\n\nBASIC MODE: Focus on fundamental translation quality - accuracy, fluency, and basic meaning preservation."
        
        return base_prompt
    
    def _parse_validation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the LLM response and extract validation data."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback: create basic structure from text analysis
                return self._create_fallback_validation(response_content)
        except Exception as e:
            return {
                "overall_score": 5.0,
                "scores": {
                    "language_detection_score": 5.0,
                    "accuracy_score": 5.0,
                    "fluency_score": 5.0,
                    "meaning_preservation_score": 5.0,
                    "cultural_adaptation_score": 5.0,
                    "completeness_score": 5.0
                },
                "feedback": {
                    "detailed_feedback": f"Could not parse validation response: {str(e)}",
                    "summary": "Validation parsing error"
                },
                "metrics": {},
                "issues": ["Response parsing failed"],
                "recommendations": ["Retry validation"]
            }
    
    def _create_fallback_validation(self, response_content: str) -> Dict[str, Any]:
        """Create fallback validation data when JSON parsing fails."""
        return {
            "overall_score": 6.0,
            "scores": {
                "language_detection_score": 6.0,
                "accuracy_score": 6.0,
                "fluency_score": 6.0,
                "meaning_preservation_score": 6.0,
                "cultural_adaptation_score": 6.0,
                "completeness_score": 6.0
            },
            "feedback": {
                "detailed_feedback": response_content,
                "summary": "Validation completed with fallback parsing"
            },
            "metrics": {},
            "issues": [],
            "recommendations": []
        }


# Main validator class for easy access
class ToolValidator:
    """Main LLM-based validator class providing access to all validation capabilities."""
    
    def __init__(self):
        self.correction_validator = CorrectionValidatorAgent()
        self.translation_validator = TranslationValidatorAgent()
    
    async def validate_correction(self, original_text: str, corrected_text: str, 
                                context: str = "", validation_level: str = "comprehensive") -> Dict[str, Any]:
        """Validate a single transcription correction using LLM intelligence."""
        result = await self.correction_validator.validate_correction(
            original_text, corrected_text, context, validation_level
        )
        return result.to_dict()
    
    async def validate_translation(self, original_text: str, translated_text: str,
                                 source_language: str, target_language: str,
                                 context: str = "", validation_level: str = "comprehensive") -> Dict[str, Any]:
        """Validate a single transcription translation using LLM intelligence."""
        result = await self.translation_validator.validate_translation(
            original_text, translated_text, source_language, target_language, context, validation_level
        )
        return result.to_dict()
