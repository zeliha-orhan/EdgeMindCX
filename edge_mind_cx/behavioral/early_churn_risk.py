"""
Early churn risk analysis module for EdgeMindCX project.

Detects early churn risk signals from audio and text features:
- Long silences
- Increasing pitch/energy (frustration indicators)
- Speech rate irregularities
- Negative word density in transcripts
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from edge_mind_cx.behavioral.text_stress_analyzer import TextStressAnalyzer

logger = logging.getLogger(__name__)


class EarlyChurnRiskAnalyzer:
    """
    Analyzes early churn risk from call center interactions.
    
    Detects warning signals that indicate customer frustration
    and potential early call termination.
    """
    
    # Risk thresholds
    LONG_SILENCE_THRESHOLD = 3.0  # seconds
    PITCH_INCREASE_THRESHOLD = 0.15  # 15% increase
    ENERGY_INCREASE_THRESHOLD = 0.20  # 20% increase
    SPEECH_RATE_VARIANCE_THRESHOLD = 0.3  # coefficient of variation
    NEGATIVE_WORD_DENSITY_THRESHOLD = 0.15  # 15% negative words
    
    def __init__(
        self,
        long_silence_threshold: float = 3.0,
        pitch_increase_threshold: float = 0.15,
        energy_increase_threshold: float = 0.20,
        speech_rate_variance_threshold: float = 0.3,
        negative_word_density_threshold: float = 0.15,
    ) -> None:
        """
        Initialize early churn risk analyzer.
        
        Args:
            long_silence_threshold: Threshold for long silence detection (seconds).
                                  Default is 3.0.
            pitch_increase_threshold: Threshold for pitch increase detection (ratio).
                                    Default is 0.15 (15%).
            energy_increase_threshold: Threshold for energy increase detection (ratio).
                                      Default is 0.20 (20%).
            speech_rate_variance_threshold: Threshold for speech rate irregularity.
                                           Default is 0.3 (coefficient of variation).
            negative_word_density_threshold: Threshold for negative word density.
                                            Default is 0.15 (15%).
        """
        self.long_silence_threshold = long_silence_threshold
        self.pitch_increase_threshold = pitch_increase_threshold
        self.energy_increase_threshold = energy_increase_threshold
        self.speech_rate_variance_threshold = speech_rate_variance_threshold
        self.negative_word_density_threshold = negative_word_density_threshold
        
        # Initialize text stress analyzer for negative word detection
        self.text_stress_analyzer = TextStressAnalyzer()
    
    def analyze_call(
        self,
        call_id: str,
        audio_features_path: str | Path,
        diarization_path: str | Path,
        transcription_path: Optional[str | Path] = None,
        customer_speaker_id: str = "SPEAKER_01",
    ) -> Dict[str, any]:
        """
        Analyze early churn risk for a call.
        
        Args:
            call_id: Call identifier.
            audio_features_path: Path to audio features CSV file.
            diarization_path: Path to diarization JSON file.
            transcription_path: Optional path to transcription JSON file.
            customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
        
        Returns:
            Dictionary containing:
            - call_id: Call identifier
            - risk_level: Risk level (low/medium/high)
            - risk_score: Risk score (0-100)
            - risk_factors: Individual risk factor scores
            - signals: Detected risk signals
            - explanation: Human-readable explanation
        """
        logger.info(f"Analyzing early churn risk for call: {call_id}")
        
        # Load data
        audio_features_df = pd.read_csv(audio_features_path)
        with open(diarization_path, "r", encoding="utf-8") as f:
            diarization_data = json.load(f)
        
        # Get customer segments
        aligned_segments = diarization_data.get("aligned_segments", [])
        customer_segments = [
            seg for seg in aligned_segments
            if seg.get("speaker_id") == customer_speaker_id
        ]
        
        if len(customer_segments) == 0:
            logger.warning(f"No customer segments found for call {call_id}")
            return self._empty_result(call_id)
        
        # Get customer audio features
        customer_features = audio_features_df[
            audio_features_df["speaker_id"] == customer_speaker_id
        ].copy()
        
        if len(customer_features) == 0:
            logger.warning(f"No customer audio features found for call {call_id}")
            return self._empty_result(call_id)
        
        # Analyze risk factors
        silence_risk = self._analyze_long_silences(customer_segments, customer_features)
        pitch_risk = self._analyze_pitch_increase(customer_features)
        energy_risk = self._analyze_energy_increase(customer_features)
        speech_rate_risk = self._analyze_speech_rate_irregularity(customer_features)
        negative_word_risk = self._analyze_negative_words(
            call_id, transcription_path, customer_segments
        )
        
        # Calculate overall risk score
        risk_factors = {
            "long_silences": silence_risk,
            "pitch_increase": pitch_risk,
            "energy_increase": energy_risk,
            "speech_rate_irregularity": speech_rate_risk,
            "negative_word_density": negative_word_risk,
        }
        
        # Weighted risk score
        risk_score = (
            0.25 * silence_risk["risk_score"]
            + 0.20 * pitch_risk["risk_score"]
            + 0.20 * energy_risk["risk_score"]
            + 0.15 * speech_rate_risk["risk_score"]
            + 0.20 * negative_word_risk["risk_score"]
        )
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Collect risk signals
        signals = self._collect_risk_signals(risk_factors)
        
        # Generate explanation
        explanation = self._generate_explanation(risk_level, risk_score, risk_factors, signals)
        
        result = {
            "call_id": call_id,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "risk_factors": risk_factors,
            "signals": signals,
            "explanation": explanation,
        }
        
        logger.info(f"Early churn risk: {risk_level} (score: {risk_score:.2f})")
        
        return result
    
    def _analyze_long_silences(
        self,
        customer_segments: List[Dict[str, any]],
        customer_features: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze long silence periods (frustration indicator).
        
        Args:
            customer_segments: Customer speech segments.
            customer_features: Customer audio features.
        
        Returns:
            Dictionary with risk score and details.
        """
        if len(customer_segments) < 2:
            return {"risk_score": 0.0, "long_silences": [], "max_silence": 0.0}
        
        # Calculate silence periods between segments
        segments_sorted = sorted(customer_segments, key=lambda x: x["start_time"])
        long_silences = []
        
        for i in range(len(segments_sorted) - 1):
            current_end = segments_sorted[i]["end_time"]
            next_start = segments_sorted[i + 1]["start_time"]
            silence_duration = next_start - current_end
            
            if silence_duration >= self.long_silence_threshold:
                long_silences.append({
                    "start": current_end,
                    "end": next_start,
                    "duration": silence_duration,
                })
        
        max_silence = max([s["duration"] for s in long_silences], default=0.0)
        
        # Risk score based on number and duration of long silences
        num_long_silences = len(long_silences)
        risk_score = min(100.0, (num_long_silences * 20.0) + (max_silence * 10.0))
        
        return {
            "risk_score": round(risk_score, 2),
            "long_silences": long_silences,
            "num_long_silences": num_long_silences,
            "max_silence": round(max_silence, 2),
        }
    
    def _analyze_pitch_increase(
        self,
        customer_features: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze increasing pitch trend (frustration indicator).
        
        Args:
            customer_features: Customer audio features over time.
        
        Returns:
            Dictionary with risk score and details.
        """
        if "pitch_mean" not in customer_features.columns or len(customer_features) < 2:
            return {"risk_score": 0.0, "pitch_trend": 0.0, "pitch_increase": 0.0}
        
        pitch_values = customer_features["pitch_mean"].values
        pitch_values = pitch_values[pitch_values > 0]  # Filter out zeros
        
        if len(pitch_values) < 2:
            return {"risk_score": 0.0, "pitch_trend": 0.0, "pitch_increase": 0.0}
        
        # Calculate trend
        x = np.arange(len(pitch_values))
        slope, _, _, _, _ = stats.linregress(x, pitch_values)
        
        # Calculate increase ratio
        initial_pitch = pitch_values[0]
        final_pitch = pitch_values[-1]
        
        if initial_pitch > 0:
            pitch_increase = (final_pitch - initial_pitch) / initial_pitch
        else:
            pitch_increase = 0.0
        
        # Risk score: positive slope and increase above threshold
        if pitch_increase >= self.pitch_increase_threshold:
            risk_score = min(100.0, (pitch_increase / self.pitch_increase_threshold) * 50.0)
        elif slope > 0:
            risk_score = min(50.0, (slope / np.mean(pitch_values)) * 100.0)
        else:
            risk_score = 0.0
        
        return {
            "risk_score": round(risk_score, 2),
            "pitch_trend": round(float(slope), 4),
            "pitch_increase": round(pitch_increase, 4),
            "initial_pitch": round(float(initial_pitch), 2),
            "final_pitch": round(float(final_pitch), 2),
        }
    
    def _analyze_energy_increase(
        self,
        customer_features: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze increasing energy trend (frustration/anger indicator).
        
        Args:
            customer_features: Customer audio features over time.
        
        Returns:
            Dictionary with risk score and details.
        """
        if "energy_mean" not in customer_features.columns or len(customer_features) < 2:
            return {"risk_score": 0.0, "energy_trend": 0.0, "energy_increase": 0.0}
        
        energy_values = customer_features["energy_mean"].values
        energy_values = energy_values[energy_values > 0]  # Filter out zeros
        
        if len(energy_values) < 2:
            return {"risk_score": 0.0, "energy_trend": 0.0, "energy_increase": 0.0}
        
        # Calculate trend
        x = np.arange(len(energy_values))
        slope, _, _, _, _ = stats.linregress(x, energy_values)
        
        # Calculate increase ratio
        initial_energy = energy_values[0]
        final_energy = energy_values[-1]
        
        if initial_energy > 0:
            energy_increase = (final_energy - initial_energy) / initial_energy
        else:
            energy_increase = 0.0
        
        # Risk score: positive slope and increase above threshold
        if energy_increase >= self.energy_increase_threshold:
            risk_score = min(100.0, (energy_increase / self.energy_increase_threshold) * 50.0)
        elif slope > 0:
            risk_score = min(50.0, (slope / np.mean(energy_values)) * 100.0)
        else:
            risk_score = 0.0
        
        return {
            "risk_score": round(risk_score, 2),
            "energy_trend": round(float(slope), 4),
            "energy_increase": round(energy_increase, 4),
            "initial_energy": round(float(initial_energy), 4),
            "final_energy": round(float(final_energy), 4),
        }
    
    def _analyze_speech_rate_irregularity(
        self,
        customer_features: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze speech rate irregularity (uncertainty/frustration indicator).
        
        Args:
            customer_features: Customer audio features over time.
        
        Returns:
            Dictionary with risk score and details.
        """
        # Calculate speech rates from duration
        if "duration" not in customer_features.columns or len(customer_features) < 3:
            return {"risk_score": 0.0, "speech_rate_variance": 0.0, "coefficient_of_variation": 0.0}
        
        durations = customer_features["duration"].values
        durations = durations[durations > 0]
        
        if len(durations) < 3:
            return {"risk_score": 0.0, "speech_rate_variance": 0.0, "coefficient_of_variation": 0.0}
        
        # Approximate speech rate (inverse of duration)
        speech_rates = 1.0 / (durations + 0.1)
        
        # Calculate coefficient of variation
        mean_rate = np.mean(speech_rates)
        std_rate = np.std(speech_rates)
        
        if mean_rate > 0:
            coefficient_of_variation = std_rate / mean_rate
        else:
            coefficient_of_variation = 0.0
        
        # Risk score: high variance indicates irregularity
        if coefficient_of_variation >= self.speech_rate_variance_threshold:
            risk_score = min(100.0, (coefficient_of_variation / self.speech_rate_variance_threshold) * 100.0)
        else:
            risk_score = (coefficient_of_variation / self.speech_rate_variance_threshold) * 50.0
        
        return {
            "risk_score": round(risk_score, 2),
            "speech_rate_variance": round(float(std_rate), 4),
            "coefficient_of_variation": round(coefficient_of_variation, 4),
            "mean_speech_rate": round(float(mean_rate), 4),
        }
    
    def _analyze_negative_words(
        self,
        call_id: str,
        transcription_path: Optional[str | Path],
        customer_segments: List[Dict[str, any]],
    ) -> Dict[str, any]:
        """
        Analyze negative word density in customer transcript.
        
        Args:
            call_id: Call identifier.
            transcription_path: Path to transcription JSON file.
            customer_segments: Customer speech segments.
        
        Returns:
            Dictionary with risk score and details.
        """
        if transcription_path is None:
            # Try to find transcription file
            transcription_path = Path("data/processed/transcripts") / call_id / "transcript.json"
        
        transcription_path = Path(transcription_path)
        
        if not transcription_path.exists():
            logger.warning(f"Transcription file not found: {transcription_path}")
            return {"risk_score": 0.0, "negative_word_density": 0.0, "total_words": 0}
        
        # Load transcription
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)
        
        # Extract customer text
        customer_texts = []
        for seg in customer_segments:
            # Find matching transcription segment
            for trans_seg in transcription_data.get("segments", []):
                if (abs(trans_seg.get("start", 0) - seg["start_time"]) < 0.5 and
                    abs(trans_seg.get("end", 0) - seg["end_time"]) < 0.5):
                    customer_texts.append(trans_seg.get("text", ""))
                    break
        
        if not customer_texts:
            return {"risk_score": 0.0, "negative_word_density": 0.0, "total_words": 0}
        
        # Combine customer text
        customer_text = " ".join(customer_texts)
        
        # Analyze text stress
        stress_result = self.text_stress_analyzer.analyze(customer_text)
        negative_score = stress_result.get("negativity_score", 0.0)
        
        # Count words
        words = customer_text.split()
        total_words = len(words)
        
        # Risk score based on negativity
        if negative_score >= self.negative_word_density_threshold:
            risk_score = min(100.0, (negative_score / self.negative_word_density_threshold) * 100.0)
        else:
            risk_score = (negative_score / self.negative_word_density_threshold) * 50.0
        
        return {
            "risk_score": round(risk_score, 2),
            "negative_word_density": round(negative_score, 4),
            "total_words": total_words,
            "text_stress_score": round(stress_result.get("text_stress_score", 0.0), 4),
        }
    
    def _collect_risk_signals(self, risk_factors: Dict[str, Dict[str, any]]) -> List[str]:
        """
        Collect detected risk signals.
        
        Args:
            risk_factors: Dictionary of risk factor analyses.
        
        Returns:
            List of risk signal strings.
        """
        signals = []
        
        # Long silences
        if risk_factors["long_silences"]["risk_score"] > 50:
            signals.append(f"Multiple long silences detected ({risk_factors['long_silences']['num_long_silences']} instances)")
        
        # Pitch increase
        if risk_factors["pitch_increase"]["risk_score"] > 50:
            signals.append(f"Significant pitch increase detected ({risk_factors['pitch_increase']['pitch_increase']*100:.1f}%)")
        
        # Energy increase
        if risk_factors["energy_increase"]["risk_score"] > 50:
            signals.append(f"Significant energy increase detected ({risk_factors['energy_increase']['energy_increase']*100:.1f}%)")
        
        # Speech rate irregularity
        if risk_factors["speech_rate_irregularity"]["risk_score"] > 50:
            signals.append("Irregular speech rate pattern detected")
        
        # Negative words
        if risk_factors["negative_word_density"]["risk_score"] > 50:
            signals.append(f"High negative word density ({risk_factors['negative_word_density']['negative_word_density']*100:.1f}%)")
        
        if not signals:
            signals.append("No significant risk signals detected")
        
        return signals
    
    def _generate_explanation(
        self,
        risk_level: str,
        risk_score: float,
        risk_factors: Dict[str, Dict[str, any]],
        signals: List[str],
    ) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            risk_level: Risk level (low/medium/high).
            risk_score: Risk score (0-100).
            risk_factors: Risk factor analyses.
            signals: Detected risk signals.
        
        Returns:
            Explanation string.
        """
        explanation = f"Early churn risk: {risk_level.upper()} (Score: {risk_score:.1f}/100). "
        
        if risk_level == "high":
            explanation += "Strong indicators of customer frustration detected. "
        elif risk_level == "medium":
            explanation += "Moderate indicators of potential customer dissatisfaction. "
        else:
            explanation += "Low risk indicators. Customer appears engaged. "
        
        explanation += "Key signals: " + "; ".join(signals[:3]) + "."
        
        return explanation
    
    def _empty_result(self, call_id: str) -> Dict[str, any]:
        """Return empty result for insufficient data."""
        return {
            "call_id": call_id,
            "risk_level": "unknown",
            "risk_score": 0.0,
            "risk_factors": {},
            "signals": ["Insufficient data for analysis"],
            "explanation": "Insufficient data to assess early churn risk.",
        }


def analyze_early_churn_risk(
    call_id: str,
    audio_features_path: str | Path,
    diarization_path: str | Path,
    transcription_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    customer_speaker_id: str = "SPEAKER_01",
) -> Dict[str, any]:
    """
    Convenience function to analyze early churn risk for a call.
    
    Args:
        call_id: Call identifier.
        audio_features_path: Path to audio features CSV file.
        diarization_path: Path to diarization JSON file.
        transcription_path: Optional path to transcription JSON file.
        output_path: Optional path to save JSON results.
        customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
    
    Returns:
        Dictionary containing early churn risk analysis results.
    
    Example:
        >>> result = analyze_early_churn_risk(
        ...     call_id="call_abc123",
        ...     audio_features_path="data/processed/audio_features_call_abc123.csv",
        ...     diarization_path="data/processed/transcripts/call_abc123/diarization.json"
        ... )
    """
    analyzer = EarlyChurnRiskAnalyzer()
    
    result = analyzer.analyze_call(
        call_id=call_id,
        audio_features_path=audio_features_path,
        diarization_path=diarization_path,
        transcription_path=transcription_path,
        customer_speaker_id=customer_speaker_id,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved early churn risk analysis to: {output_path}")
    
    return result
