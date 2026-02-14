"""
openSMILE eGeMAPS feature extraction module for EdgeMindCX project.

This module extracts eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)
features using openSMILE for stress and emotion analysis in call center audio.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    import opensmile
except ImportError:
    opensmile = None
    logging.warning(
        "opensmile-python not installed. Install with: pip install opensmile-python"
    )

if TYPE_CHECKING:
    # For type hints only
    from opensmile import FeatureLevel, FeatureSet

logger = logging.getLogger(__name__)


class eGeMAPSExtractor:
    """
    Extracts eGeMAPS features using openSMILE.
    
    eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) is a
    standardized feature set designed for emotion and stress recognition.
    It includes 88 acoustic features covering:
    - Fundamental frequency (F0) and formants
    - Energy and loudness
    - Spectral characteristics
    - Voice quality
    - Timing features
    """
    
    def __init__(
        self,
        feature_set: Optional["FeatureSet"] = None,
        feature_level: Optional["FeatureLevel"] = None,
    ) -> None:
        """
        Initialize eGeMAPS feature extractor.
        
        Args:
            feature_set: openSMILE feature set. Defaults to eGeMAPSv02.
            feature_level: Feature level (Functionals or LLDs).
                          Defaults to Functionals (aggregated features).
        
        Raises:
            ImportError: If opensmile-python is not installed.
        """
        if opensmile is None:
            raise ImportError(
                "opensmile-python is required. Install with: pip install opensmile-python"
            )
        
        # Default to eGeMAPSv02 feature set (88 features)
        if feature_set is None and opensmile is not None:
            feature_set = opensmile.FeatureSet.eGeMAPSv02
        
        # Default to Functionals (aggregated statistics over entire audio)
        if feature_level is None and opensmile is not None:
            feature_level = opensmile.FeatureLevel.Functionals
        
        self.feature_set = feature_set
        self.feature_level = feature_level
        
        # Initialize openSMILE
        self.smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=feature_level,
        )
        
        logger.info(
            f"Initialized eGeMAPS extractor: {feature_set}, {feature_level}"
        )
    
    def extract_from_path(
        self,
        audio_path: Union[str, Path],
        return_dict: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        Extract eGeMAPS features from audio file path.
        
        Args:
            audio_path: Path to audio file.
            return_dict: If True, return as dictionary. If False, return DataFrame.
                        Default is False (DataFrame).
        
        Returns:
            DataFrame or dictionary containing eGeMAPS features.
            DataFrame has one row with feature names as columns.
            Dictionary has feature names as keys.
        
        Raises:
            FileNotFoundError: If audio file does not exist.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.debug(f"Extracting eGeMAPS features from: {audio_path}")
        
        try:
            # Extract features using openSMILE
            features = self.smile.process_file(str(audio_path))
            
            # Convert to desired format
            if return_dict:
                return self._to_dict(features)
            else:
                return features
        
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def _to_dict(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Convert DataFrame features to dictionary.
        
        Args:
            features: DataFrame with features.
        
        Returns:
            Dictionary with feature names as keys and values as floats.
        """
        # Flatten multi-level column names if present
        if isinstance(features.columns, pd.MultiIndex):
            feature_names = [
                "_".join(col).strip() if col[1] else col[0]
                for col in features.columns.values
            ]
        else:
            feature_names = features.columns.tolist()
        
        # Get values (should be single row)
        values = features.iloc[0].values
        
        return dict(zip(feature_names, values))
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of eGeMAPS features.
        
        Returns:
            Dictionary mapping feature names to descriptions.
        """
        return {
            # F0 (Fundamental Frequency) Features - Pitch characteristics
            "F0semitoneFrom27.5Hz_sma3nz_amean": "Mean F0 in semitones (relative to 27.5Hz), indicates average pitch",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "F0 standard deviation (normalized), indicates pitch variability",
            "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "20th percentile F0, lower pitch range indicator",
            "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": "50th percentile F0 (median), typical pitch level",
            "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "80th percentile F0, higher pitch range indicator",
            "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "F0 range (0-2), pitch variation measure",
            "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": "Mean rising F0 slope, pitch increase rate",
            "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": "Mean falling F0 slope, pitch decrease rate",
            
            # Loudness Features - Perceived volume and energy
            "loudness_sma3_amean": "Mean loudness, overall perceived volume level",
            "loudness_sma3_stddevNorm": "Loudness standard deviation, volume variability",
            "loudness_sma3_percentile20.0": "20th percentile loudness, quiet segments indicator",
            "loudness_sma3_percentile50.0": "50th percentile loudness (median), typical volume",
            "loudness_sma3_percentile80.0": "80th percentile loudness, loud segments indicator",
            "loudness_sma3_pctlrange0-2": "Loudness range, volume variation measure",
            "loudness_sma3_meanRisingSlope": "Mean rising loudness slope, volume increase rate",
            "loudness_sma3_meanFallingSlope": "Mean falling loudness slope, volume decrease rate",
            
            # Spectral Features - Frequency domain characteristics
            "spectralFlux_sma3_amean": "Mean spectral flux, spectral change rate (indicates speech activity)",
            "spectralFlux_sma3_stddevNorm": "Spectral flux variability",
            "mfcc1_sma3_amean": "Mean MFCC 1, spectral envelope shape (related to voice quality)",
            "mfcc1_sma3_stddevNorm": "MFCC 1 variability",
            "mfcc2_sma3_amean": "Mean MFCC 2, spectral characteristics",
            "mfcc2_sma3_stddevNorm": "MFCC 2 variability",
            "mfcc3_sma3_amean": "Mean MFCC 3, spectral characteristics",
            "mfcc3_sma3_stddevNorm": "MFCC 3 variability",
            "mfcc4_sma3_amean": "Mean MFCC 4, spectral characteristics",
            "mfcc4_sma3_stddevNorm": "MFCC 4 variability",
            
            # Formant Features - Vocal tract characteristics
            "F1frequency_sma3nz_amean": "Mean F1 frequency, first formant (vowel quality indicator)",
            "F1frequency_sma3nz_stddevNorm": "F1 variability",
            "F1bandwidth_sma3nz_amean": "Mean F1 bandwidth, formant sharpness",
            "F1bandwidth_sma3nz_stddevNorm": "F1 bandwidth variability",
            "F2frequency_sma3nz_amean": "Mean F2 frequency, second formant (vowel quality)",
            "F2frequency_sma3nz_stddevNorm": "F2 variability",
            "F2bandwidth_sma3nz_amean": "Mean F2 bandwidth",
            "F2bandwidth_sma3nz_stddevNorm": "F2 bandwidth variability",
            "F3frequency_sma3nz_amean": "Mean F3 frequency, third formant",
            "F3frequency_sma3nz_stddevNorm": "F3 variability",
            "F3bandwidth_sma3nz_amean": "Mean F3 bandwidth",
            "F3bandwidth_sma3nz_stddevNorm": "F3 bandwidth variability",
            
            # Jitter and Shimmer - Voice quality indicators (stress markers)
            "jitterLocal_sma3nz_amean": "Mean local jitter, pitch period variability (stress indicator)",
            "jitterLocal_sma3nz_stddevNorm": "Jitter variability",
            "shimmerLocal_sma3nz_amean": "Mean local shimmer, amplitude variability (stress indicator)",
            "shimmerLocal_sma3nz_stddevNorm": "Shimmer variability",
            
            # Harmonic-to-Noise Ratio - Voice quality
            "HNRdBACF_sma3nz_amean": "Mean HNR (Harmonic-to-Noise Ratio), voice quality indicator",
            "HNRdBACF_sma3nz_stddevNorm": "HNR variability",
            
            # Alpha Ratio - Spectral balance
            "alphaRatio_sma3_amean": "Mean alpha ratio, low-to-high frequency energy balance",
            "alphaRatio_sma3_stddevNorm": "Alpha ratio variability",
            
            # Hammarberg Index - Spectral characteristics
            "hammarbergIndex_sma3_amean": "Mean Hammarberg index, spectral peak location",
            "hammarbergIndex_sma3_stddevNorm": "Hammarberg index variability",
            
            # Spectral Slope - Frequency characteristics
            "slope0-500_sma3_amean": "Mean spectral slope (0-500 Hz), low frequency energy",
            "slope0-500_sma3_stddevNorm": "Spectral slope (0-500 Hz) variability",
            "slope500-1500_sma3_amean": "Mean spectral slope (500-1500 Hz), mid frequency energy",
            "slope500-1500_sma3_stddevNorm": "Spectral slope (500-1500 Hz) variability",
            
            # Formant 1-3 Amplitude - Vocal tract resonance
            "F1amplitudeLogRelF0_sma3nz_amean": "Mean F1 amplitude (log, relative to F0)",
            "F1amplitudeLogRelF0_sma3nz_stddevNorm": "F1 amplitude variability",
            "F2amplitudeLogRelF0_sma3nz_amean": "Mean F2 amplitude (log, relative to F0)",
            "F2amplitudeLogRelF0_sma3nz_stddevNorm": "F2 amplitude variability",
            "F3amplitudeLogRelF0_sma3nz_amean": "Mean F3 amplitude (log, relative to F0)",
            "F3amplitudeLogRelF0_sma3nz_stddevNorm": "F3 amplitude variability",
            
            # Timing Features - Speech rhythm and rate
            "rate_sma3nz_amean": "Mean speech rate, articulation speed",
            "rate_sma3nz_stddevNorm": "Speech rate variability",
            "pauses_sma3nz_amean": "Mean pause rate, silence frequency",
            "pauses_sma3nz_stddevNorm": "Pause rate variability",
            
            # Additional Spectral Features
            "spectralCentroid_sma3_amean": "Mean spectral centroid, brightness of sound",
            "spectralCentroid_sma3_stddevNorm": "Spectral centroid variability",
            "spectralVariance_sma3_amean": "Mean spectral variance, frequency spread",
            "spectralVariance_sma3_stddevNorm": "Spectral variance variability",
            "spectralSkewness_sma3_amean": "Mean spectral skewness, frequency distribution asymmetry",
            "spectralSkewness_sma3_stddevNorm": "Spectral skewness variability",
            "spectralKurtosis_sma3_amean": "Mean spectral kurtosis, frequency distribution peakedness",
            "spectralKurtosis_sma3_stddevNorm": "Spectral kurtosis variability",
            "spectralRollOff25.0_sma3_amean": "Mean spectral rolloff (25%), frequency below which 25% energy is contained",
            "spectralRollOff25.0_sma3_stddevNorm": "Spectral rolloff (25%) variability",
            "spectralRollOff50.0_sma3_amean": "Mean spectral rolloff (50%), frequency below which 50% energy is contained",
            "spectralRollOff50.0_sma3_stddevNorm": "Spectral rolloff (50%) variability",
            "spectralRollOff75.0_sma3_amean": "Mean spectral rolloff (75%), frequency below which 75% energy is contained",
            "spectralRollOff75.0_sma3_stddevNorm": "Spectral rolloff (75%) variability",
            "spectralRollOff90.0_sma3_amean": "Mean spectral rolloff (90%), frequency below which 90% energy is contained",
            "spectralRollOff90.0_sma3_stddevNorm": "Spectral rolloff (90%) variability",
            
            # Zero Crossing Rate - Speech activity indicator
            "zcr_sma3_amean": "Mean zero crossing rate, speech activity and noise indicator",
            "zcr_sma3_stddevNorm": "Zero crossing rate variability",
        }


def extract_egemaps_features(
    audio_path: Union[str, Path],
    return_dict: bool = False,
    feature_set: Optional["FeatureSet"] = None,
    feature_level: Optional["FeatureLevel"] = None,
) -> Union[pd.DataFrame, Dict[str, float]]:
    """
    Wrapper function to extract eGeMAPS features from audio file.
    
    Convenience function for quick feature extraction without instantiating
    the eGeMAPSExtractor class.
    
    Args:
        audio_path: Path to audio file.
        return_dict: If True, return as dictionary. If False, return DataFrame.
                    Default is False (DataFrame).
        feature_set: openSMILE feature set. Defaults to eGeMAPSv02.
        feature_level: Feature level (Functionals or LLDs).
                      Defaults to Functionals (aggregated features).
    
    Returns:
        DataFrame or dictionary containing eGeMAPS features.
    
    Example:
        >>> features_df = extract_egemaps_features("path/to/audio.wav")
        >>> features_dict = extract_egemaps_features("path/to/audio.wav", return_dict=True)
    """
    extractor = eGeMAPSExtractor(
        feature_set=feature_set,
        feature_level=feature_level,
    )
    
    return extractor.extract_from_path(
        audio_path=audio_path,
        return_dict=return_dict,
    )
