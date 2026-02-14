"""
Audio-first data loader module for EdgeMindCX project.

This module provides audio-first architecture for loading call center audio files
without requiring CSV metadata. Processes .wav files directly from a directory.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioDataLoader:
    """
    Audio-first data loader for call center audio files.
    
    Loads .wav files directly from a directory without requiring CSV metadata.
    Generates unique call_id for each audio file.
    """
    
    def __init__(
        self,
        audio_dir: str | Path = "data/raw/audio/call_center",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> None:
        """
        Initialize audio-first data loader.
        
        Args:
            audio_dir: Directory containing .wav audio files.
                     Default is "data/raw/audio/call_center".
            sample_rate: Target sample rate for audio loading. Default is 16000 Hz.
            mono: Whether to convert audio to mono. Default is True.
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.mono = mono
        
        self._validate_directory()
    
    def _validate_directory(self) -> None:
        """Validate that audio directory exists."""
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}"
            )
        logger.info(f"Audio directory validated: {self.audio_dir}")
    
    def _load_audio(
        self,
        audio_path: Path,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Tuple of (waveform, sample_rate).
            waveform: Audio signal as numpy array.
            sample_rate: Actual sample rate of loaded audio.
        
        Raises:
            FileNotFoundError: If audio file does not exist.
            librosa.LibrosaError: If audio file cannot be loaded.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            waveform, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=self.mono,
            )
            logger.debug(
                f"Loaded audio: {audio_path.name}, "
                f"shape: {waveform.shape}, sr: {sr}"
            )
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def _generate_call_id(self, audio_path: Path) -> str:
        """
        Generate unique call ID for audio file.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Unique call ID string.
        """
        # Use UUID4 for unique ID, but also include filename for traceability
        call_id = f"call_{uuid.uuid4().hex[:8]}_{audio_path.stem}"
        return call_id
    
    def load_audio_file(
        self,
        audio_path: str | Path,
        call_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Load a single audio file.
        
        Args:
            audio_path: Path to audio file (can be relative to audio_dir or absolute).
            call_id: Optional call ID. If None, will be generated automatically.
        
        Returns:
            Dictionary containing:
            - call_id: Unique call identifier
            - waveform: Audio signal as numpy array
            - sample_rate: Sample rate of the audio
            - audio_path: Path to the audio file
            - filename: Audio filename
        """
        audio_path = Path(audio_path)
        
        # If relative path, resolve relative to audio_dir
        if not audio_path.is_absolute():
            audio_path = self.audio_dir / audio_path
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate call_id if not provided
        if call_id is None:
            call_id = self._generate_call_id(audio_path)
        
        # Load audio
        waveform, sample_rate = self._load_audio(audio_path)
        
        return {
            "call_id": call_id,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "audio_path": str(audio_path),
            "filename": audio_path.name,
        }
    
    def iter_audio_files(
        self,
        pattern: str = "*.wav",
        recursive: bool = False,
    ) -> Iterator[Dict[str, any]]:
        """
        Iterator over all audio files in the directory.
        
        Args:
            pattern: File pattern to match. Default is "*.wav".
            recursive: Whether to search recursively. Default is False.
        
        Yields:
            Dictionary containing audio data (see load_audio_file for structure).
        """
        if recursive:
            audio_files = list(self.audio_dir.rglob(pattern))
        else:
            audio_files = list(self.audio_dir.glob(pattern))
        
        logger.info(f"Found {len(audio_files)} audio files matching pattern '{pattern}'")
        
        for audio_path in audio_files:
            try:
                sample = self.load_audio_file(audio_path)
                yield sample
            except Exception as e:
                logger.error(f"Error loading audio file {audio_path}: {e}")
                continue
    
    def get_audio_files(
        self,
        pattern: str = "*.wav",
        recursive: bool = False,
    ) -> list[Path]:
        """
        Get list of audio files in the directory.
        
        Args:
            pattern: File pattern to match. Default is "*.wav".
            recursive: Whether to search recursively. Default is False.
        
        Returns:
            List of Path objects for audio files.
        """
        if recursive:
            audio_files = list(self.audio_dir.rglob(pattern))
        else:
            audio_files = list(self.audio_dir.glob(pattern))
        
        return audio_files
