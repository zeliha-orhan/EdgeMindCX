"""
Stress score fusion module for EdgeMindCX project.

This module combines audio-based and text-based stress signals into a unified
stress score using weighted averaging. Integrates:
- Audio features (librosa-based)
- openSMILE eGeMAPS features
- NLP text stress score
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StressScoreFusion:
    """
    Combines multiple stress signals into a unified stress score.
    
    Integrates audio features, openSMILE features, and NLP text analysis
    using configurable weighted averaging.
    """
    
    def __init__(
        self,
        audio_weight: float = 0.3,
        opensmile_weight: float = 0.4,
        nlp_weight: float = 0.3,
        normalize_weights: bool = True,
    ) -> None:
        """
        Initialize stress score fusion.
        
        Args:
            audio_weight: Weight for audio features (librosa-based) stress score.
                        Default is 0.3.
            opensmile_weight: Weight for openSMILE eGeMAPS features stress score.
                             Default is 0.4.
            nlp_weight: Weight for NLP text-based stress score.
                       Default is 0.3.
            normalize_weights: If True, automatically normalize weights to sum to 1.0.
                             Default is True.
        
        Raises:
            ValueError: If weights are negative or don't sum to 1.0 (when normalize=False).
        """
        self.audio_weight = audio_weight
        self.opensmile_weight = opensmile_weight
        self.nlp_weight = nlp_weight
        
        # Validate and normalize weights
        if normalize_weights:
            total = audio_weight + opensmile_weight + nlp_weight
            if total > 0:
                self.audio_weight = audio_weight / total
                self.opensmile_weight = opensmile_weight / total
                self.nlp_weight = nlp_weight / total
                logger.info(
                    f"Weights normalized. Final weights: "
                    f"audio={self.audio_weight:.2f}, "
                    f"opensmile={self.opensmile_weight:.2f}, "
                    f"nlp={self.nlp_weight:.2f}"
                )
            else:
                raise ValueError("Sum of weights must be greater than 0")
        else:
            total = audio_weight + opensmile_weight + nlp_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0 (current sum: {total}). "
                    f"Set normalize_weights=True to auto-normalize."
                )
            if any(w < 0 for w in [audio_weight, opensmile_weight, nlp_weight]):
                raise ValueError("Weights cannot be negative")
        
        logger.info(
            f"StressScoreFusion initialized with weights: "
            f"audio={self.audio_weight:.2f}, "
            f"opensmile={self.opensmile_weight:.2f}, "
            f"nlp={self.nlp_weight:.2f}"
        )
    
    def compute_final_score(
        self,
        audio_features: Optional[Dict[str, float]] = None,
        opensmile_features: Optional[Union[Dict[str, float], pd.DataFrame]] = None,
        nlp_stress_score: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute final unified stress score from multiple sources.
        
        Args:
            audio_features: Dictionary of audio features from AudioFeatureExtractor.
                          Expected keys: pitch_variance, silence_ratio, speech_rate, etc.
            opensmile_features: Dictionary or DataFrame of eGeMAPS features from openSMILE.
                              Can be dict or DataFrame (will be converted).
            nlp_stress_score: Text-based stress score (0-1) from TextStressAnalyzer.
        
        Returns:
            Dictionary containing:
            - stress_score: Final unified stress score (0-1)
            - audio_stress_score: Audio-based stress score (0-1)
            - opensmile_stress_score: openSMILE-based stress score (0-1)
            - nlp_stress_score: NLP-based stress score (0-1)
            - component_scores: Detailed breakdown of component scores
        """
        # Compute individual stress scores
        audio_stress = self._compute_audio_stress(audio_features)
        opensmile_stress = self._compute_opensmile_stress(opensmile_features)
        nlp_stress = self._normalize_nlp_score(nlp_stress_score)
        
        # Weighted combination
        final_score = (
            self.audio_weight * audio_stress
            + self.opensmile_weight * opensmile_stress
            + self.nlp_weight * nlp_stress
        )
        
        # Clamp to [0, 1]
        final_score = float(np.clip(final_score, 0.0, 1.0))
        
        return {
            "stress_score": final_score,
            "audio_stress_score": audio_stress,
            "opensmile_stress_score": opensmile_stress,
            "nlp_stress_score": nlp_stress,
            "component_scores": {
                "audio": {
                    "score": audio_stress,
                    "weight": self.audio_weight,
                    "contribution": self.audio_weight * audio_stress,
                },
                "opensmile": {
                    "score": opensmile_stress,
                    "weight": self.opensmile_weight,
                    "contribution": self.opensmile_weight * opensmile_stress,
                },
                "nlp": {
                    "score": nlp_stress,
                    "weight": self.nlp_weight,
                    "contribution": self.nlp_weight * nlp_stress,
                },
            },
        }
    
    def _compute_audio_stress(
        self,
        audio_features: Optional[Dict[str, float]],
    ) -> float:
        """
        Compute stress score from librosa audio features.
        
        Stress indicators from audio:
        - High pitch variance (unstable pitch = stress)
        - High silence ratio (hesitation = stress)
        - Low speech rate (slow speech = stress/uncertainty)
        - High energy variance (inconsistent volume = stress)
        
        Args:
            audio_features: Dictionary of audio features.
        
        Returns:
            Audio-based stress score (0-1).
        """
        if audio_features is None or len(audio_features) == 0:
            logger.warning("No audio features provided, using default score 0.5")
            return 0.5
        
        # Extract relevant features
        pitch_variance = audio_features.get("pitch_variance", 0.0)
        silence_ratio = audio_features.get("silence_ratio", 0.0)
        speech_rate = audio_features.get("speech_rate", 1.0)
        energy_std = audio_features.get("energy_rms_std", 0.0)
        
        # Normalize features to [0, 1] range
        # Pitch variance: higher = more stress (normalize assuming max ~10000 Hz²)
        pitch_variance_norm = min(pitch_variance / 10000.0, 1.0) if pitch_variance > 0 else 0.0
        
        # Silence ratio: higher = more stress (already 0-1)
        silence_ratio_norm = silence_ratio
        
        # Speech rate: lower = more stress (invert: 1 - speech_rate)
        speech_rate_inv = 1.0 - speech_rate
        
        # Energy std: higher = more stress (normalize assuming max ~0.5)
        energy_std_norm = min(energy_std / 0.5, 1.0) if energy_std > 0 else 0.0
        
        # Combine with weights (tunable based on domain knowledge)
        audio_stress = (
            0.3 * pitch_variance_norm
            + 0.3 * silence_ratio_norm
            + 0.2 * speech_rate_inv
            + 0.2 * energy_std_norm
        )
        
        return float(np.clip(audio_stress, 0.0, 1.0))
    
    def _compute_opensmile_stress(
        self,
        opensmile_features: Optional[Union[Dict[str, float], pd.DataFrame]],
    ) -> float:
        """
        Compute stress score from openSMILE eGeMAPS features.
        
        Stress indicators from eGeMAPS:
        - High jitter (voice instability = stress)
        - High shimmer (amplitude instability = stress)
        - High F0 variability (pitch instability = stress)
        - Low HNR (voice quality degradation = stress)
        - High negative emotion scores
        
        Args:
            opensmile_features: Dictionary or DataFrame of eGeMAPS features.
        
        Returns:
            openSMILE-based stress score (0-1).
        """
        if opensmile_features is None:
            logger.warning("No openSMILE features provided, using default score 0.5")
            return 0.5
        
        # Convert DataFrame to dict if needed
        if isinstance(opensmile_features, pd.DataFrame):
            # Flatten multi-level columns if present
            if isinstance(opensmile_features.columns, pd.MultiIndex):
                feature_names = [
                    "_".join(col).strip() if col[1] else col[0]
                    for col in opensmile_features.columns.values
                ]
            else:
                feature_names = opensmile_features.columns.tolist()
            
            values = opensmile_features.iloc[0].values
            opensmile_features = dict(zip(feature_names, values))
        
        # Extract stress-relevant features
        # Jitter (local jitter - voice instability)
        jitter_mean = opensmile_features.get(
            "jitterLocal_sma3nz_amean", 0.0
        )
        jitter_std = opensmile_features.get(
            "jitterLocal_sma3nz_stddevNorm", 0.0
        )
        
        # Shimmer (local shimmer - amplitude instability)
        shimmer_mean = opensmile_features.get(
            "shimmerLocal_sma3nz_amean", 0.0
        )
        shimmer_std = opensmile_features.get(
            "shimmerLocal_sma3nz_stddevNorm", 0.0
        )
        
        # F0 variability (pitch instability)
        f0_std = opensmile_features.get(
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.0
        )
        
        # HNR (Harmonic-to-Noise Ratio - lower = more stress)
        hnr_mean = opensmile_features.get("HNRdBACF_sma3nz_amean", 20.0)
        
        # Normalize features to [0, 1]
        # Jitter: higher = more stress (normalize assuming max ~0.05)
        jitter_norm = min(jitter_mean / 0.05, 1.0) if jitter_mean > 0 else 0.0
        jitter_std_norm = min(jitter_std / 0.1, 1.0) if jitter_std > 0 else 0.0
        
        # Shimmer: higher = more stress (normalize assuming max ~0.5)
        shimmer_norm = min(shimmer_mean / 0.5, 1.0) if shimmer_mean > 0 else 0.0
        shimmer_std_norm = min(shimmer_std / 0.2, 1.0) if shimmer_std > 0 else 0.0
        
        # F0 std: higher = more stress (normalize assuming max ~10 semitones)
        f0_std_norm = min(f0_std / 10.0, 1.0) if f0_std > 0 else 0.0
        
        # HNR: lower = more stress (invert and normalize: assume range 0-30 dB)
        hnr_norm = 1.0 - min(hnr_mean / 30.0, 1.0) if hnr_mean > 0 else 1.0
        
        # Combine with weights
        opensmile_stress = (
            0.25 * jitter_norm
            + 0.15 * jitter_std_norm
            + 0.25 * shimmer_norm
            + 0.15 * shimmer_std_norm
            + 0.15 * f0_std_norm
            + 0.05 * hnr_norm
        )
        
        return float(np.clip(opensmile_stress, 0.0, 1.0))
    
    def _normalize_nlp_score(self, nlp_stress_score: Optional[float]) -> float:
        """
        Normalize NLP stress score to [0, 1] range.
        
        Args:
            nlp_stress_score: NLP stress score (should already be 0-1).
        
        Returns:
            Normalized NLP stress score (0-1).
        """
        if nlp_stress_score is None:
            logger.warning("No NLP stress score provided, using default score 0.5")
            return 0.5
        
        # Clamp to [0, 1]
        return float(np.clip(nlp_stress_score, 0.0, 1.0))
    
    def update_weights(
        self,
        audio_weight: Optional[float] = None,
        opensmile_weight: Optional[float] = None,
        nlp_weight: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        """
        Update fusion weights dynamically.
        
        Allows runtime adjustment of weights for different use cases or
        calibration based on validation data.
        
        Args:
            audio_weight: New weight for audio features (if provided).
            opensmile_weight: New weight for openSMILE features (if provided).
            nlp_weight: New weight for NLP features (if provided).
            normalize: If True, normalize weights to sum to 1.0.
        
        Example:
            >>> fusion = StressScoreFusion()
            >>> fusion.update_weights(audio_weight=0.5, nlp_weight=0.5)
        """
        if audio_weight is not None:
            self.audio_weight = audio_weight
        if opensmile_weight is not None:
            self.opensmile_weight = opensmile_weight
        if nlp_weight is not None:
            self.nlp_weight = nlp_weight
        
        if normalize:
            total = self.audio_weight + self.opensmile_weight + self.nlp_weight
            if total > 0:
                self.audio_weight = self.audio_weight / total
                self.opensmile_weight = self.opensmile_weight / total
                self.nlp_weight = self.nlp_weight / total
        
        logger.info(
            f"Weights updated: audio={self.audio_weight:.2f}, "
            f"opensmile={self.opensmile_weight:.2f}, "
            f"nlp={self.nlp_weight:.2f}"
        )


def compute_stress_score(
    audio_features: Optional[Dict[str, float]] = None,
    opensmile_features: Optional[Union[Dict[str, float], pd.DataFrame]] = None,
    nlp_stress_score: Optional[float] = None,
    audio_weight: float = 0.3,
    opensmile_weight: float = 0.4,
    nlp_weight: float = 0.3,
) -> float:
    """
    Wrapper function to quickly compute final stress score.
    
    Convenience function for simple stress score computation.
    
    Args:
        audio_features: Dictionary of audio features.
        opensmile_features: Dictionary or DataFrame of eGeMAPS features.
        nlp_stress_score: Text-based stress score (0-1).
        audio_weight: Weight for audio features. Default is 0.3.
        opensmile_weight: Weight for openSMILE features. Default is 0.4.
        nlp_weight: Weight for NLP features. Default is 0.3.
    
    Returns:
        Final unified stress score (0-1).
    
    Example:
        >>> score = compute_stress_score(
        ...     audio_features=audio_feats,
        ...     opensmile_features=egemaps_feats,
        ...     nlp_stress_score=0.75
        ... )
    """
    fusion = StressScoreFusion(
        audio_weight=audio_weight,
        opensmile_weight=opensmile_weight,
        nlp_weight=nlp_weight,
    )
    
    result = fusion.compute_final_score(
        audio_features=audio_features,
        opensmile_features=opensmile_features,
        nlp_stress_score=nlp_stress_score,
    )
    
    return result["stress_score"]
