"""
Audio feature extraction module for EdgeMindCX project.

This module extracts behavioral features from audio waveforms for CX analysis.
Features include pitch, energy, speech rate, and silence detection.
"""

import logging
from typing import Dict

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    Extracts behavioral features from audio waveforms.
    
    Features extracted:
    - Pitch (average, variance)
    - Energy (RMS)
    - Speech rate (approximate)
    - Silence duration and ratio
    """
    
    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> None:
        """
        Initialize audio feature extractor.
        
        Args:
            silence_threshold_db: Threshold in dB for silence detection.
                                Default is -40.0 dB.
            frame_length: Frame length for analysis. Default is 2048 samples.
            hop_length: Hop length for analysis. Default is 512 samples.
        """
        self.silence_threshold_db = silence_threshold_db
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract_features(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Extract all behavioral features from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array (1D for mono).
            sample_rate: Sample rate of the audio in Hz.
        
        Returns:
            Dictionary containing extracted features:
            - pitch_mean: Average pitch in Hz
            - pitch_variance: Variance of pitch in Hz²
            - energy_rms_mean: Mean RMS energy
            - energy_rms_std: Standard deviation of RMS energy
            - speech_rate: Approximate speech rate (ratio of speech to total duration)
            - silence_duration: Total silence duration in seconds
            - silence_ratio: Ratio of silence to total duration
            - total_duration: Total audio duration in seconds
        """
        if waveform.size == 0:
            logger.warning("Empty waveform provided")
            return self._empty_features()
        
        # Ensure mono audio
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)
        
        # Calculate total duration
        total_duration = len(waveform) / sample_rate
        
        # Extract individual features
        pitch_features = self._extract_pitch(waveform, sample_rate)
        energy_features = self._extract_energy(waveform)
        silence_features = self._extract_silence(waveform, sample_rate)
        speech_rate = self._calculate_speech_rate(
            waveform, sample_rate, silence_features["silence_ratio"]
        )
        
        # Combine all features
        features = {
            **pitch_features,
            **energy_features,
            **silence_features,
            "speech_rate": speech_rate,
            "total_duration": total_duration,
        }
        
        logger.debug(f"Extracted {len(features)} features from audio")
        return features
    
    def _extract_pitch(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Extract pitch features using librosa's pyin algorithm.
        
        pyin (Probabilistic YIN) provides robust pitch estimation even in
        noisy conditions, which is important for call center audio.
        
        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate in Hz.
        
        Returns:
            Dictionary with pitch_mean and pitch_variance.
        """
        try:
            # Use pyin for robust pitch estimation
            # fmin and fmax are typical human voice range
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz("C2"),  # ~65 Hz (low male voice)
                fmax=librosa.note_to_hz("C7"),  # ~2093 Hz (high female voice)
                sr=sample_rate,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
            
            # Filter out unvoiced segments (NaN values)
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) == 0:
                logger.warning("No voiced segments found for pitch estimation")
                return {"pitch_mean": 0.0, "pitch_variance": 0.0}
            
            pitch_mean = float(np.mean(f0_voiced))
            pitch_variance = float(np.var(f0_voiced))
            
            return {
                "pitch_mean": pitch_mean,
                "pitch_variance": pitch_variance,
            }
        except Exception as e:
            logger.error(f"Error extracting pitch: {e}")
            return {"pitch_mean": 0.0, "pitch_variance": 0.0}
    
    def _extract_energy(
        self,
        waveform: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extract RMS energy features.
        
        RMS (Root Mean Square) energy represents the overall loudness
        of the audio signal, which can indicate speaker engagement level.
        
        Args:
            waveform: Audio waveform.
        
        Returns:
            Dictionary with energy_rms_mean and energy_rms_std.
        """
        try:
            # Calculate RMS energy per frame
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )[0]
            
            if len(rms) == 0:
                return {"energy_rms_mean": 0.0, "energy_rms_std": 0.0}
            
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            
            return {
                "energy_rms_mean": energy_mean,
                "energy_rms_std": energy_std,
            }
        except Exception as e:
            logger.error(f"Error extracting energy: {e}")
            return {"energy_rms_mean": 0.0, "energy_rms_std": 0.0}
    
    def _extract_silence(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Extract silence features.
        
        Detects silent segments based on energy threshold and calculates
        silence duration and ratio. This helps identify pauses, hesitations,
        and overall speech flow.
        
        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate in Hz.
        
        Returns:
            Dictionary with silence_duration and silence_ratio.
        """
        try:
            # Calculate RMS energy per frame
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )[0]
            
            if len(rms) == 0:
                return {"silence_duration": 0.0, "silence_ratio": 0.0}
            
            # Convert RMS to dB
            rms_db = librosa.power_to_db(rms**2, ref=np.max(rms**2))
            
            # Identify silent frames (below threshold)
            silent_frames = rms_db < self.silence_threshold_db
            
            # Calculate silence duration
            # Each frame represents hop_length samples
            frame_duration = self.hop_length / sample_rate
            silence_duration = float(np.sum(silent_frames) * frame_duration)
            
            # Calculate silence ratio
            total_duration = len(waveform) / sample_rate
            silence_ratio = silence_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                "silence_duration": silence_duration,
                "silence_ratio": silence_ratio,
            }
        except Exception as e:
            logger.error(f"Error extracting silence: {e}")
            return {"silence_duration": 0.0, "silence_ratio": 0.0}
    
    def _calculate_speech_rate(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        silence_ratio: float,
    ) -> float:
        """
        Calculate approximate speech rate.
        
        Speech rate is estimated as the ratio of non-silent segments
        to total duration. This provides an approximation of how much
        of the audio contains actual speech vs. pauses.
        
        Higher speech rate indicates more continuous speech,
        lower rate indicates more pauses and hesitations.
        
        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate in Hz.
            silence_ratio: Ratio of silence (from silence extraction).
        
        Returns:
            Speech rate as a float (0.0 to 1.0, where 1.0 = continuous speech).
        """
        # Speech rate is the complement of silence ratio
        # If 30% is silence, then 70% is speech
        speech_rate = 1.0 - silence_ratio
        
        # Clamp to valid range
        speech_rate = max(0.0, min(1.0, speech_rate))
        
        return float(speech_rate)
    
    def _empty_features(self) -> Dict[str, float]:
        """
        Return empty feature dictionary for error cases.
        
        Returns:
            Dictionary with all features set to 0.0.
        """
        return {
            "pitch_mean": 0.0,
            "pitch_variance": 0.0,
            "energy_rms_mean": 0.0,
            "energy_rms_std": 0.0,
            "speech_rate": 0.0,
            "silence_duration": 0.0,
            "silence_ratio": 0.0,
            "total_duration": 0.0,
        }
