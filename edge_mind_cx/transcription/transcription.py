"""
Transcription module for EdgeMindCX project.

This module provides functionality to transcribe audio using local Whisper models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import whisper

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """
    Container for transcription results.
    
    Attributes:
        text: Full transcribed text.
        segments: List of segments with timing information.
    """
    
    def __init__(
        self,
        text: str,
        segments: List[Dict[str, Union[str, float]]],
    ) -> None:
        """
        Initialize transcription result.
        
        Args:
            text: Full transcribed text.
            segments: List of segment dictionaries containing:
                     - text: Segment text
                     - start_time: Start time in seconds
                     - end_time: End time in seconds
        """
        self.text = text
        self.segments = segments
    
    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary with 'text' and 'segments' keys.
        """
        return {
            "text": self.text,
            "segments": self.segments,
        }


class WhisperTranscriber:
    """
    Local Whisper-based transcription service.
    
    Uses open-source Whisper models (small or medium) for speech-to-text
    transcription without requiring API keys.
    """
    
    # Supported model sizes
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size. Options: tiny, base, small, medium, large.
                       Default is 'small'. Recommended: 'small' or 'medium'.
            device: Device to run model on ('cpu' or 'cuda'). If None, auto-detects.
            language: Language code (e.g., 'en', 'tr'). If None, auto-detects.
        
        Raises:
            ValueError: If model_size is not supported.
        """
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model size: {model_size}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        
        self.model_size = model_size
        self.device = device
        self.language = language
        
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size, device=device)
        logger.info(f"Whisper model loaded successfully")
    
    def transcribe_from_path(
        self,
        audio_path: Union[str, Path],
        **whisper_kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio from file path.
        
        Args:
            audio_path: Path to audio file.
            **whisper_kwargs: Additional arguments to pass to Whisper's transcribe method.
                            Common options: temperature, beam_size, best_of, etc.
        
        Returns:
            TranscriptionResult containing text and segments with timing.
        
        Raises:
            FileNotFoundError: If audio file does not exist.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.debug(f"Transcribing audio from path: {audio_path}")
        
        # Prepare transcription options
        transcribe_options = {}
        if self.language:
            transcribe_options["language"] = self.language
        
        # Merge with additional kwargs
        transcribe_options.update(whisper_kwargs)
        
        # Transcribe using Whisper
        result = self.model.transcribe(
            str(audio_path),
            **transcribe_options,
        )
        
        return self._process_whisper_result(result)
    
    def transcribe_from_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        **whisper_kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio from waveform array.
        
        Args:
            waveform: Audio waveform as numpy array (1D for mono, 2D for stereo).
            sample_rate: Sample rate of the waveform. Default is 16000 Hz.
            **whisper_kwargs: Additional arguments to pass to Whisper's transcribe method.
                            Common options: temperature, beam_size, best_of, etc.
        
        Returns:
            TranscriptionResult containing text and segments with timing.
        
        Raises:
            ValueError: If waveform is empty or invalid.
        """
        if waveform.size == 0:
            raise ValueError("Waveform is empty")
        
        logger.debug(
            f"Transcribing waveform: shape={waveform.shape}, "
            f"sample_rate={sample_rate}"
        )
        
        # Prepare transcription options
        transcribe_options = {}
        if self.language:
            transcribe_options["language"] = self.language
        
        # Merge with additional kwargs
        transcribe_options.update(whisper_kwargs)
        
        # Transcribe using Whisper
        # Whisper expects float32 audio in range [-1, 1], 16 kHz (DecodingOptions does not accept sample_rate)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Normalize if needed (Whisper handles this internally, but ensure range)
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            waveform = np.clip(waveform, -1.0, 1.0)
        
        # Resample to 16 kHz if needed (Whisper expects 16 kHz for numpy input)
        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=16000
            )
        
        result = self.model.transcribe(waveform, **transcribe_options)
        
        return self._process_whisper_result(result)
    
    def _process_whisper_result(
        self,
        whisper_result: Dict,
    ) -> TranscriptionResult:
        """
        Process Whisper result into TranscriptionResult format.
        
        Args:
            whisper_result: Raw result dictionary from Whisper.
        
        Returns:
            TranscriptionResult with formatted text and segments.
        """
        # Extract full text
        text = whisper_result.get("text", "").strip()
        
        # Extract segments with timing
        segments = []
        for segment in whisper_result.get("segments", []):
            segments.append({
                "text": segment.get("text", "").strip(),
                "start_time": segment.get("start", 0.0),
                "end_time": segment.get("end", 0.0),
            })
        
        return TranscriptionResult(text=text, segments=segments)
    
    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
        **whisper_kwargs,
    ) -> TranscriptionResult:
        """
        Unified transcription method that accepts either path or waveform.
        
        Args:
            audio_input: Audio file path (str/Path) or waveform array (np.ndarray).
            sample_rate: Sample rate (required if audio_input is waveform).
            **whisper_kwargs: Additional arguments to pass to Whisper.
        
        Returns:
            TranscriptionResult containing text and segments with timing.
        
        Raises:
            ValueError: If input type is invalid or sample_rate is missing for waveform.
        """
        if isinstance(audio_input, (str, Path)):
            return self.transcribe_from_path(audio_input, **whisper_kwargs)
        elif isinstance(audio_input, np.ndarray):
            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required when audio_input is a waveform"
                )
            return self.transcribe_from_waveform(
                audio_input, sample_rate=sample_rate, **whisper_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported audio_input type: {type(audio_input)}. "
                f"Expected str, Path, or np.ndarray"
            )
