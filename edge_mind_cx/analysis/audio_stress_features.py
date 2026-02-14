"""
Audio-based stress and emotion signal extraction module for EdgeMindCX project.

Extracts speaker-aware audio features using librosa and openSMILE for stress
and emotion analysis. Features are normalized and saved per speaker.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from edge_mind_cx.analysis.audio_features import AudioFeatureExtractor
from edge_mind_cx.analysis.opensmile_features import extract_egemaps_features

logger = logging.getLogger(__name__)


class AudioStressFeatureExtractor:
    """
    Extracts audio-based stress and emotion signals per speaker.
    
    Features:
    - librosa: pitch (F0), energy, tempo
    - openSMILE: prosodic features, arousal-related features
    - Speaker-aware segmentation
    - Normalized scores
    """
    
    def __init__(
        self,
        silence_threshold_db: float = -40.0,
    ) -> None:
        """
        Initialize audio stress feature extractor.
        
        Args:
            silence_threshold_db: Threshold in dB for silence detection.
                                Default is -40.0 dB.
        """
        self.audio_feature_extractor = AudioFeatureExtractor(
            silence_threshold_db=silence_threshold_db
        )
    
    def extract_librosa_features(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Extract librosa-based features: pitch (F0), energy, tempo.
        
        Args:
            waveform: Audio waveform as numpy array.
            sample_rate: Sample rate in Hz.
        
        Returns:
            Dictionary containing:
            - pitch_mean: Mean F0 in Hz
            - pitch_std: Standard deviation of F0
            - pitch_range: F0 range (max - min)
            - energy_mean: Mean RMS energy
            - energy_std: Standard deviation of RMS energy
            - tempo: Estimated tempo in BPM
        """
        if waveform.size == 0:
            return self._empty_librosa_features()
        
        # Ensure mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)
        
        features = {}
        
        # Pitch (F0) features
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 0:
                features["pitch_mean"] = float(np.mean(f0_voiced))
                features["pitch_std"] = float(np.std(f0_voiced))
                features["pitch_range"] = float(np.max(f0_voiced) - np.min(f0_voiced))
            else:
                features.update({
                    "pitch_mean": 0.0,
                    "pitch_std": 0.0,
                    "pitch_range": 0.0,
                })
        except Exception as e:
            logger.warning(f"Error extracting pitch: {e}")
            features.update({
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_range": 0.0,
            })
        
        # Energy features
        try:
            rms = librosa.feature.rms(y=waveform)[0]
            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))
        except Exception as e:
            logger.warning(f"Error extracting energy: {e}")
            features.update({
                "energy_mean": 0.0,
                "energy_std": 0.0,
            })
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
            features["tempo"] = float(tempo)
        except Exception as e:
            logger.warning(f"Error extracting tempo: {e}")
            features["tempo"] = 0.0
        
        return features
    
    def extract_opensmile_features(
        self,
        audio_path: str | Path,
    ) -> Dict[str, float]:
        """
        Extract openSMILE prosodic and arousal-related features.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Dictionary containing prosodic and arousal features.
        """
        try:
            opensmile_features = extract_egemaps_features(
                audio_path=audio_path,
                return_dict=True,
            )
            
            # Select prosodic and arousal-related features
            prosodic_features = {}
            
            # Prosodic features (F0, loudness, formants)
            prosodic_keys = [
                "F0semitoneFrom27.5Hz_sma3nz_amean",
                "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
                "loudness_sma3_amean",
                "loudness_sma3_stddevNorm",
                "F1frequency_sma3nz_amean",
                "F2frequency_sma3nz_amean",
            ]
            
            # Arousal-related features (jitter, shimmer, HNR, energy)
            arousal_keys = [
                "jitterLocal_sma3nz_amean",
                "shimmerLocal_sma3nz_amean",
                "HNRdBACF_sma3nz_amean",
                "spectralFlux_sma3_amean",
                "spectralCentroid_sma3_amean",
            ]
            
            for key in prosodic_keys + arousal_keys:
                if key in opensmile_features:
                    prosodic_features[key] = opensmile_features[key]
            
            return prosodic_features
            
        except Exception as e:
            logger.warning(f"Error extracting openSMILE features: {e}")
            return {}
    
    def extract_speaker_segment(
        self,
        audio_path: str | Path,
        start_time: float,
        end_time: float,
    ) -> tuple[np.ndarray, int]:
        """
        Extract audio segment for a specific time range.
        
        Args:
            audio_path: Path to audio file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
        
        Returns:
            Tuple of (waveform, sample_rate).
        """
        try:
            # Load full audio
            waveform, sample_rate = librosa.load(str(audio_path), sr=None)
            
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment
            segment = waveform[start_sample:end_sample]
            
            return segment, sample_rate
            
        except Exception as e:
            logger.error(f"Error extracting segment: {e}")
            raise
    
    def process_call(
        self,
        call_id: str,
        audio_path: str | Path,
        diarization_path: str | Path,
    ) -> pd.DataFrame:
        """
        Process a call and extract speaker-aware audio features.
        
        Args:
            call_id: Call identifier.
            audio_path: Path to audio file.
            diarization_path: Path to diarization JSON file.
        
        Returns:
            DataFrame with features per speaker.
        """
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)
        
        logger.info(f"Processing audio features for call: {call_id}")
        
        # Load diarization data
        with open(diarization_path, "r", encoding="utf-8") as f:
            diarization_data = json.load(f)
        
        diarization_segments = diarization_data.get("diarization_segments", [])
        
        if not diarization_segments:
            logger.warning(f"No diarization segments found for call {call_id}")
            return pd.DataFrame()
        
        all_features = []
        
        # Process each speaker segment
        for segment in diarization_segments:
            speaker_id = segment["speaker_id"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            logger.debug(
                f"Processing segment: {speaker_id} "
                f"({start_time:.2f}s - {end_time:.2f}s)"
            )
            
            try:
                # Extract audio segment
                segment_waveform, sample_rate = self.extract_speaker_segment(
                    audio_path=audio_path,
                    start_time=start_time,
                    end_time=end_time,
                )
                
                # Extract librosa features
                librosa_features = self.extract_librosa_features(
                    waveform=segment_waveform,
                    sample_rate=sample_rate,
                )
                
                # Extract openSMILE features for this segment
                # Save segment temporarily for openSMILE processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    sf.write(str(tmp_path), segment_waveform, sample_rate)
                    
                    try:
                        opensmile_features = self.extract_opensmile_features(tmp_path)
                    except Exception as e:
                        logger.warning(f"Error extracting openSMILE features for segment: {e}")
                        opensmile_features = {}
                    finally:
                        # Clean up temporary file
                        if tmp_path.exists():
                            tmp_path.unlink()
                
                # Create feature row
                feature_row = {
                    "call_id": call_id,
                    "speaker_id": speaker_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    **librosa_features,
                    **opensmile_features,
                }
                
                all_features.append(feature_row)
                
            except Exception as e:
                logger.error(f"Error processing segment {speaker_id}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        if len(df) > 0:
            # Aggregate by speaker
            df_aggregated = self._aggregate_by_speaker(df)
            
            # Normalize features
            df_aggregated = self._normalize_features(df_aggregated)
            
            logger.info(f"Extracted features for {len(df_aggregated)} speakers")
            
            return df_aggregated
        else:
            return df
    
    def _aggregate_by_speaker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features by speaker.
        
        Args:
            df: DataFrame with segment-level features.
        
        Returns:
            DataFrame with speaker-level aggregated features.
        """
        if "speaker_id" not in df.columns:
            return df
        
        # Group by speaker and aggregate
        agg_dict = {
            "call_id": "first",
            "start_time": "min",
            "end_time": "max",
            "duration": "sum",
        }
        
        # Aggregate numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["start_time", "end_time", "duration"]:
                agg_dict[col] = "mean"  # Use mean for most features
        
        df_aggregated = df.groupby("speaker_id").agg(agg_dict).reset_index()
        
        # Calculate total duration per speaker
        df_aggregated["total_duration"] = df_aggregated["end_time"] - df_aggregated["start_time"]
        
        return df_aggregated
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range.
        
        Args:
            df: DataFrame with features.
        
        Returns:
            DataFrame with normalized features.
        """
        # Features to normalize
        normalize_cols = [
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "energy_mean",
            "energy_std",
            "tempo",
        ]
        
        df_normalized = df.copy()
        
        for col in normalize_cols:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_max > col_min:
                    df_normalized[f"{col}_normalized"] = (
                        (df[col] - col_min) / (col_max - col_min)
                    )
                else:
                    df_normalized[f"{col}_normalized"] = 0.0
        
        return df_normalized
    
    def _empty_librosa_features(self) -> Dict[str, float]:
        """Return empty librosa features dictionary."""
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "tempo": 0.0,
        }
    
    def save_features(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """
        Save features to CSV file.
        
        Args:
            df: DataFrame with features.
            output_path: Path to output CSV file.
        
        Returns:
            Path to saved CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved audio features to: {output_path}")
        
        return output_path


def extract_audio_stress_features(
    call_id: str,
    audio_path: str | Path,
    diarization_path: str | Path,
    output_dir: Optional[str | Path] = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to extract audio-based stress features for a call.
    
    Args:
        call_id: Call identifier.
        audio_path: Path to audio file.
        diarization_path: Path to diarization JSON file.
        output_dir: Output directory. If None, uses 'data/processed'.
        save_results: Whether to save results to CSV. Default is True.
    
    Returns:
        DataFrame with features per speaker segment.
    
    Example:
        >>> df = extract_audio_stress_features(
        ...     call_id="call_abc123",
        ...     audio_path="data/raw/audio/call_center/call1.wav",
        ...     diarization_path="data/processed/transcripts/call_abc123/diarization.json"
        ... )
    """
    extractor = AudioStressFeatureExtractor()
    
    df = extractor.process_call(
        call_id=call_id,
        audio_path=audio_path,
        diarization_path=diarization_path,
    )
    
    if save_results and len(df) > 0:
        if output_dir is None:
            output_dir = Path("data/processed")
        else:
            output_dir = Path(output_dir)
        
        output_path = output_dir / f"audio_features_{call_id}.csv"
        extractor.save_features(df, output_path)
    
    return df
