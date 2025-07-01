"""Speech recognition using OpenAI Whisper."""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, BinaryIO

import torch
import whisper
from pydub import AudioSegment

from ..core.models import ASRResult

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based speech recognition transcriber."""

    def __init__(
            self,
            model_name: str = "base",
            device: Optional[str] = None,
            download_root: Optional[str] = None
    ):
        """Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, or auto-detect)
            download_root: Directory to store model files
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.download_root = download_root
        self.model = None

        logger.info(f"Initializing Whisper transcriber with model '{model_name}' on device '{self.device}'")

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=self.download_root
            )
            logger.info("Whisper model loaded successfully")

    def _prepare_audio(self, audio_input: Union[str, Path, BinaryIO, bytes]) -> str:
        """Prepare audio for transcription by converting to a supported format.
        
        Args:
            audio_input: Audio file path, file object, or bytes
            
        Returns:
            Path to the prepared audio file
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if audio_path.exists():
                supported_formats = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg'}
                if audio_path.suffix.lower() in supported_formats:
                    return str(audio_path)

        try:
            if isinstance(audio_input, (str, Path)):
                audio = AudioSegment.from_file(audio_input)
            elif isinstance(audio_input, bytes):
                audio = AudioSegment.from_file(io.BytesIO(audio_input))
            else:
                audio = AudioSegment.from_file(audio_input)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio.export(temp_file.name, format='wav')
                return temp_file.name

        except Exception as e:
            logger.error(f"Failed to prepare audio: {e}")
            raise ValueError(f"Could not process audio input: {e}")

    async def transcribe_file(
            self,
            audio_path: Union[str, Path],
            language: Optional[str] = None,
            task: str = "transcribe",
            **kwargs
    ) -> ASRResult:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'es'). None for auto-detection
            task: 'transcribe' or 'translate'
            **kwargs: Additional parameters for whisper.transcribe()
            
        Returns:
            ASRResult with transcription and metadata
        """
        self._load_model()

        try:
            prepared_path = self._prepare_audio(audio_path)
            temp_file_created = prepared_path != str(audio_path)

            logger.info(f"Transcribing audio file: {audio_path}")

            result = self.model.transcribe(
                prepared_path,
                language=language,
                task=task,
                **kwargs
            )

            if temp_file_created:
                try:
                    os.unlink(prepared_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {prepared_path}: {e}")

            segments = []
            if 'segments' in result:
                segments = [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'],
                        'confidence': seg.get('confidence', None)
                    }
                    for seg in result['segments']
                ]

            confidence = None
            if segments and all('confidence' in seg for seg in result.get('segments', [])):
                confidences = [seg['confidence'] for seg in result['segments'] if 'confidence' in seg]
                confidence = sum(confidences) / len(confidences) if confidences else None

            asr_result = ASRResult(
                text=result['text'].strip(),
                confidence=confidence,
                language=result.get('language'),
                timestamps=segments,
                metadata={
                    'model': self.model_name,
                    'device': self.device,
                    'task': task,
                    'audio_duration': len(AudioSegment.from_file(audio_path)) / 1000.0,  # seconds
                    'segments_count': len(segments)
                }
            )

            logger.info(f"Transcription completed. Text length: {len(asr_result.text)} characters")
            return asr_result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    async def transcribe_bytes(
            self,
            audio_bytes: bytes,
            language: Optional[str] = None,
            task: str = "transcribe",
            **kwargs
    ) -> ASRResult:
        """Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            language: Language code (e.g., 'en', 'es'). None for auto-detection
            task: 'transcribe' or 'translate'
            **kwargs: Additional parameters for whisper.transcribe()
            
        Returns:
            ASRResult with transcription and metadata
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            result = await self.transcribe_file(temp_path, language, task, **kwargs)
            return result
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

    def get_available_models(self) -> list[str]:
        """Get list of available Whisper models."""
        return ["tiny", "base", "small", "medium", "large"]

    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "tr": "Turkish",
            "pl": "Polish",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "no": "Norwegian",
            "fi": "Finnish"
        }

    def estimate_processing_time(self, audio_duration_seconds: float) -> float:
        """Estimate processing time based on audio duration and model size.
        
        Args:
            audio_duration_seconds: Duration of audio in seconds
            
        Returns:
            Estimated processing time in seconds
        """
        model_factors = {
            "tiny": 0.1,
            "base": 0.2,
            "small": 0.4,
            "medium": 0.8,
            "large": 1.5
        }

        base_factor = model_factors.get(self.model_name, 0.5)

        if self.device == "cuda":
            base_factor *= 0.25

        return audio_duration_seconds * base_factor


def create_transcriber(
        model_name: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None
) -> WhisperTranscriber:
    """Create a Whisper transcriber with the specified configuration.
    
    Args:
        model_name: Whisper model to use
        device: Device to run on
        download_root: Directory to store model files
        
    Returns:
        Configured WhisperTranscriber instance
    """
    return WhisperTranscriber(
        model_name=model_name,
        device=device,
        download_root=download_root
    )
