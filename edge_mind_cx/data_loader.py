"""
Data loader module for EdgeMindCX project.

This module provides functionality to load IEMOCAP dataset metadata and audio files
for processing in the EdgeMindCX pipeline.

DEPRECATED: This module is deprecated in favor of audio-first architecture.
Use AudioDataLoader from audio_loader.py instead.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IEMOCAPDataLoader:
    """
    Production-level data loader for IEMOCAP dataset.
    
    Handles CSV metadata loading, audio file path resolution,
    file existence validation, and audio loading with librosa.
    
    .. deprecated:: 0.2.0
        This class is deprecated. Use :class:`AudioDataLoader` from
        :mod:`edge_mind_cx.audio_loader` instead for audio-first architecture.
    """
    
    def __init__(
        self,
        metadata_path: str | Path,
        audio_base_path: str | Path = "data/raw/audio",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> None:
        """
        Initialize the IEMOCAP data loader.
        
        Args:
            metadata_path: Path to the IEMOCAP metadata CSV file.
            audio_base_path: Base path for audio files. Audio paths from CSV
                           will be resolved relative to this path.
            sample_rate: Target sample rate for audio loading. Default is 16000 Hz.
            mono: Whether to convert audio to mono. Default is True.
        """
        self.metadata_path = Path(metadata_path)
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = sample_rate
        self.mono = mono
        
        self._metadata: Optional[pd.DataFrame] = None
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that metadata file exists."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )
        logger.info(f"Metadata file validated: {self.metadata_path}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and return the metadata CSV file.
        
        Returns:
            DataFrame containing IEMOCAP metadata with columns:
            session, method, gender, emotion, n_annotators, agreement, path
            
        Raises:
            FileNotFoundError: If metadata file does not exist.
            ValueError: If required columns are missing.
        """
        if self._metadata is not None:
            return self._metadata
        
        logger.info(f"Loading metadata from: {self.metadata_path}")
        self._metadata = pd.read_csv(self.metadata_path)
        
        # Validate required columns
        required_columns = [
            "session", "method", "gender", "emotion",
            "n_annotators", "agreement", "path"
        ]
        missing_columns = set(required_columns) - set(self._metadata.columns)
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns in metadata: {missing_columns}"
            )
        
        logger.info(f"Loaded {len(self._metadata)} records from metadata")
        return self._metadata
    
    def _resolve_audio_path(self, relative_path: str) -> Path:
        """
        Resolve audio file path relative to audio_base_path.
        
        Args:
            relative_path: Relative path from CSV (e.g., 
                          "Session1/sentences/wav/Ses01F_script02_1/Ses01F_script02_1_F000.wav")
        
        Returns:
            Resolved absolute Path object.
        """
        # Remove leading slash if present
        relative_path = relative_path.lstrip("/")
        resolved_path = self.audio_base_path / relative_path
        return resolved_path.resolve()
    
    def _validate_audio_file(self, audio_path: Path) -> bool:
        """
        Validate that audio file exists.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            True if file exists, False otherwise.
        """
        exists = audio_path.exists()
        if not exists:
            logger.warning(f"Audio file not found: {audio_path}")
        return exists
    
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
    
    def load_sample(
        self,
        row: pd.Series,
        validate_file: bool = True,
    ) -> Optional[Dict[str, any]]:
        """
        Load a single sample from metadata row.
        
        Args:
            row: Pandas Series representing a single row from metadata.
            validate_file: Whether to validate file existence before loading.
                          If False, will attempt to load even if file doesn't exist.
        
        Returns:
            Dictionary containing:
            - waveform: Audio signal as numpy array
            - sample_rate: Sample rate of the audio
            - emotion: Emotion label
            - gender: Gender label
            - session: Session identifier
            - agreement: Agreement score
            None if file validation fails and validate_file is True.
        """
        audio_path = self._resolve_audio_path(row["path"])
        
        if validate_file and not self._validate_audio_file(audio_path):
            return None
        
        try:
            waveform, sample_rate = self._load_audio(audio_path)
            
            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "emotion": row["emotion"],
                "gender": row["gender"],
                "session": row["session"],
                "agreement": row["agreement"],
            }
        except Exception as e:
            logger.error(f"Error loading sample for row {row.name}: {e}")
            return None
    
    def iter_samples(
        self,
        validate_file: bool = True,
        skip_missing: bool = True,
    ) -> Iterator[Dict[str, any]]:
        """
        Iterator over all samples in the metadata.
        
        Args:
            validate_file: Whether to validate file existence before loading.
            skip_missing: Whether to skip samples with missing files.
                         If False, will raise error for missing files.
        
        Yields:
            Dictionary containing sample data (see load_sample for structure).
        """
        metadata = self.load_metadata()
        
        for idx, row in metadata.iterrows():
            sample = self.load_sample(row, validate_file=validate_file)
            
            if sample is None:
                if skip_missing:
                    logger.warning(
                        f"Skipping sample at index {idx} due to missing file"
                    )
                    continue
                else:
                    raise FileNotFoundError(
                        f"Required audio file missing for row {idx}"
                    )
            
            yield sample
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics:
            - total_samples: Total number of samples in metadata
            - available_samples: Number of samples with existing audio files
            - missing_samples: Number of samples with missing audio files
            - emotion_distribution: Distribution of emotions
            - gender_distribution: Distribution of genders
            - session_distribution: Distribution across sessions
        """
        metadata = self.load_metadata()
        
        total_samples = len(metadata)
        available_samples = 0
        missing_samples = 0
        
        for _, row in metadata.iterrows():
            audio_path = self._resolve_audio_path(row["path"])
            if audio_path.exists():
                available_samples += 1
            else:
                missing_samples += 1
        
        stats = {
            "total_samples": total_samples,
            "available_samples": available_samples,
            "missing_samples": missing_samples,
            "emotion_distribution": metadata["emotion"].value_counts().to_dict(),
            "gender_distribution": metadata["gender"].value_counts().to_dict(),
            "session_distribution": metadata["session"].value_counts().to_dict(),
        }
        
        return stats
