"""
Audio ingestion module for EdgeMindCX project.

This module scans, validates, and normalizes call center audio files.
Performs automatic conversion to standard format (16kHz, mono).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioIngestion:
    """
    Audio ingestion and validation for call center audio files.
    
    Scans .wav files, validates format, and converts to standard format:
    - Sample rate: 16kHz
    - Channels: Mono
    """
    
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono
    
    def __init__(
        self,
        audio_dir: str | Path = "data/raw/audio/call_center",
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        auto_convert: bool = True,
    ) -> None:
        """
        Initialize audio ingestion.
        
        Args:
            audio_dir: Directory containing .wav audio files.
                     Default is "data/raw/audio/call_center".
            target_sample_rate: Target sample rate for conversion. Default is 16000 Hz.
            target_channels: Target number of channels (1=mono, 2=stereo). Default is 1.
            auto_convert: Whether to automatically convert files. Default is True.
        """
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.auto_convert = auto_convert
        
        self._validate_directory()
    
    def _validate_directory(self) -> None:
        """Validate that audio directory exists."""
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}"
            )
        logger.info(f"Audio directory validated: {self.audio_dir}")
    
    def _get_audio_info(
        self,
        audio_path: Path,
    ) -> Dict[str, any]:
        """
        Get audio file information without loading full audio.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Dictionary containing audio metadata:
            - duration: Duration in seconds
            - sample_rate: Sample rate in Hz
            - channels: Number of channels
            - frames: Number of audio frames
        """
        try:
            info = sf.info(str(audio_path))
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
            }
        except Exception as e:
            logger.error(f"Error reading audio info for {audio_path}: {e}")
            raise
    
    def _convert_audio(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Convert audio file to target format.
        
        Converts:
        - Sample rate to target_sample_rate (if different)
        - Channels to mono (if stereo)
        
        Args:
            audio_path: Path to input audio file.
            output_path: Path to save converted file. If None, overwrites original.
        
        Returns:
            Path to converted audio file.
        """
        if output_path is None:
            output_path = audio_path
        
        logger.debug(f"Converting audio: {audio_path.name}")
        
        try:
            # Load audio with librosa (handles resampling automatically)
            waveform, sr = librosa.load(
                str(audio_path),
                sr=self.target_sample_rate,
                mono=(self.target_channels == 1),
            )
            
            # Save converted audio
            sf.write(
                str(output_path),
                waveform,
                self.target_sample_rate,
                subtype="PCM_16",  # 16-bit PCM
            )
            
            logger.debug(f"Converted audio saved: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio {audio_path}: {e}")
            raise
    
    def scan_and_validate(
        self,
        pattern: str = "*.wav",
        recursive: bool = False,
        convert_files: Optional[bool] = None,
    ) -> List[Dict[str, any]]:
        """
        Scan directory for audio files and validate/normalize them.
        
        Args:
            pattern: File pattern to match. Default is "*.wav".
            recursive: Whether to search recursively. Default is False.
            convert_files: Whether to convert files. If None, uses auto_convert setting.
        
        Returns:
            List of validated audio dictionaries, each containing:
            - file_path: Path to audio file
            - filename: Audio filename
            - duration: Duration in seconds
            - sample_rate: Sample rate in Hz (after conversion if applicable)
            - channels: Number of channels (after conversion if applicable)
            - original_sample_rate: Original sample rate (before conversion)
            - original_channels: Original number of channels (before conversion)
            - converted: Whether file was converted
            - status: Validation status ("valid", "converted", "error")
        """
        if convert_files is None:
            convert_files = self.auto_convert
        
        # Find all audio files
        if recursive:
            audio_files = list(self.audio_dir.rglob(pattern))
        else:
            audio_files = list(self.audio_dir.glob(pattern))
        
        logger.info(f"Found {len(audio_files)} audio files matching pattern '{pattern}'")
        
        validated_audio_list = []
        
        for audio_path in audio_files:
            try:
                logger.info(f"Processing: {audio_path.name}")
                
                # Get original audio info
                original_info = self._get_audio_info(audio_path)
                
                original_sample_rate = original_info["sample_rate"]
                original_channels = original_info["channels"]
                
                # Check if conversion is needed
                needs_resample = original_sample_rate != self.target_sample_rate
                needs_mono = original_channels != self.target_channels
                needs_conversion = needs_resample or needs_mono
                
                # Convert if needed and enabled
                converted = False
                if needs_conversion and convert_files:
                    logger.info(
                        f"Converting {audio_path.name}: "
                        f"SR {original_sample_rate}Hz -> {self.target_sample_rate}Hz, "
                        f"Channels {original_channels} -> {self.target_channels}"
                    )
                    self._convert_audio(audio_path)
                    converted = True
                    
                    # Get info after conversion
                    final_info = self._get_audio_info(audio_path)
                    final_sample_rate = final_info["sample_rate"]
                    final_channels = final_info["channels"]
                else:
                    final_sample_rate = original_sample_rate
                    final_channels = original_channels
                    final_info = original_info
                
                # Determine status
                if converted:
                    status = "converted"
                elif needs_conversion and not convert_files:
                    status = "needs_conversion"
                else:
                    status = "valid"
                
                # Create validated audio entry
                validated_audio = {
                    "file_path": str(audio_path),
                    "filename": audio_path.name,
                    "duration": final_info["duration"],
                    "sample_rate": final_sample_rate,
                    "channels": final_channels,
                    "original_sample_rate": original_sample_rate,
                    "original_channels": original_channels,
                    "converted": converted,
                    "status": status,
                }
                
                validated_audio_list.append(validated_audio)
                
                # Log summary
                logger.info(
                    f"✓ {audio_path.name}: "
                    f"Duration={validated_audio['duration']:.2f}s, "
                    f"SR={final_sample_rate}Hz, "
                    f"Channels={final_channels}, "
                    f"Status={status}"
                )
                
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                
                # Add error entry
                validated_audio_list.append({
                    "file_path": str(audio_path),
                    "filename": audio_path.name,
                    "duration": None,
                    "sample_rate": None,
                    "channels": None,
                    "original_sample_rate": None,
                    "original_channels": None,
                    "converted": False,
                    "status": "error",
                    "error": str(e),
                })
                continue
        
        # Log summary statistics
        self._log_summary(validated_audio_list)
        
        return validated_audio_list
    
    def _log_summary(self, validated_audio_list: List[Dict[str, any]]) -> None:
        """
        Log summary statistics of validated audio files.
        
        Args:
            validated_audio_list: List of validated audio dictionaries.
        """
        total = len(validated_audio_list)
        valid = sum(1 for a in validated_audio_list if a["status"] == "valid")
        converted = sum(1 for a in validated_audio_list if a["status"] == "converted")
        needs_conversion = sum(
            1 for a in validated_audio_list if a["status"] == "needs_conversion"
        )
        errors = sum(1 for a in validated_audio_list if a["status"] == "error")
        
        total_duration = sum(
            a["duration"] for a in validated_audio_list if a["duration"] is not None
        )
        
        logger.info("=" * 60)
        logger.info("AUDIO INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files: {total}")
        logger.info(f"  Valid: {valid}")
        logger.info(f"  Converted: {converted}")
        logger.info(f"  Needs conversion: {needs_conversion}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        logger.info("=" * 60)


def ingest_audio_files(
    audio_dir: str | Path = "data/raw/audio/call_center",
    pattern: str = "*.wav",
    recursive: bool = False,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    auto_convert: bool = True,
) -> List[Dict[str, any]]:
    """
    Convenience function to scan and validate audio files.
    
    Args:
        audio_dir: Directory containing .wav audio files.
        pattern: File pattern to match. Default is "*.wav".
        recursive: Whether to search recursively. Default is False.
        target_sample_rate: Target sample rate for conversion. Default is 16000 Hz.
        target_channels: Target number of channels (1=mono). Default is 1.
        auto_convert: Whether to automatically convert files. Default is True.
    
    Returns:
        List of validated audio dictionaries.
    
    Example:
        >>> validated_list = ingest_audio_files(
        ...     audio_dir="data/raw/audio/call_center"
        ... )
    """
    ingestion = AudioIngestion(
        audio_dir=audio_dir,
        target_sample_rate=target_sample_rate,
        target_channels=target_channels,
        auto_convert=auto_convert,
    )
    
    return ingestion.scan_and_validate(
        pattern=pattern,
        recursive=recursive,
    )
