"""
Empathy convergence analysis module for EdgeMindCX project.

Analyzes temporal convergence of communication patterns between agent and customer
to measure empathy alignment. Tracks speech rate, pitch, and energy trends over time.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class EmpathyConvergenceAnalyzer:
    """
    Analyzes empathy convergence between agent and customer over time.
    
    Measures:
    - Speech rate alignment and convergence
    - Pitch variation alignment and convergence
    - Energy trend alignment and convergence
    - Temporal convergence patterns
    """
    
    def __init__(
        self,
        speech_rate_weight: float = 0.35,
        pitch_weight: float = 0.35,
        energy_weight: float = 0.30,
        min_segments: int = 3,
    ) -> None:
        """
        Initialize empathy convergence analyzer.
        
        Args:
            speech_rate_weight: Weight for speech rate component. Default is 0.35.
            pitch_weight: Weight for pitch component. Default is 0.35.
            energy_weight: Weight for energy component. Default is 0.30.
            min_segments: Minimum number of segments required for analysis. Default is 3.
        """
        self.speech_rate_weight = speech_rate_weight
        self.pitch_weight = pitch_weight
        self.energy_weight = energy_weight
        self.min_segments = min_segments
        
        # Normalize weights
        total = speech_rate_weight + pitch_weight + energy_weight
        if total > 0:
            self.speech_rate_weight = speech_rate_weight / total
            self.pitch_weight = pitch_weight / total
            self.energy_weight = energy_weight / total
    
    def analyze_call(
        self,
        call_id: str,
        audio_features_path: str | Path,
        diarization_path: str | Path,
        agent_speaker_id: str = "SPEAKER_00",
        customer_speaker_id: str = "SPEAKER_01",
    ) -> Dict[str, any]:
        """
        Analyze empathy convergence for a call.
        
        Args:
            call_id: Call identifier.
            audio_features_path: Path to audio features CSV file.
            diarization_path: Path to diarization JSON file.
            agent_speaker_id: Speaker ID for agent. Default is "SPEAKER_00".
            customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
        
        Returns:
            Dictionary containing:
            - call_id: Call identifier
            - empathy_alignment_score: Overall score (0-100)
            - speech_rate_alignment: Speech rate alignment score (0-100)
            - pitch_alignment: Pitch alignment score (0-100)
            - energy_alignment: Energy alignment score (0-100)
            - convergence_trend: Convergence trend analysis
            - metrics: Detailed explainable metrics
        """
        logger.info(f"Analyzing empathy convergence for call: {call_id}")
        
        # Load data
        audio_features_df = pd.read_csv(audio_features_path)
        with open(diarization_path, "r", encoding="utf-8") as f:
            diarization_data = json.load(f)
        
        # Get speaker segments with timestamps
        aligned_segments = diarization_data.get("aligned_segments", [])
        
        # Extract agent and customer data
        agent_data = self._extract_speaker_data(
            audio_features_df, aligned_segments, agent_speaker_id
        )
        customer_data = self._extract_speaker_data(
            audio_features_df, aligned_segments, customer_speaker_id
        )
        
        if len(agent_data) < self.min_segments or len(customer_data) < self.min_segments:
            logger.warning(
                f"Insufficient segments for analysis: "
                f"agent={len(agent_data)}, customer={len(customer_data)}"
            )
            return self._empty_result(call_id)
        
        # Analyze each component
        speech_rate_analysis = self._analyze_speech_rate_convergence(
            agent_data, customer_data
        )
        pitch_analysis = self._analyze_pitch_convergence(agent_data, customer_data)
        energy_analysis = self._analyze_energy_convergence(agent_data, customer_data)
        
        # Calculate overall alignment score
        alignment_score = (
            self.speech_rate_weight * speech_rate_analysis["alignment_score"]
            + self.pitch_weight * pitch_analysis["alignment_score"]
            + self.energy_weight * energy_analysis["alignment_score"]
        )
        
        # Scale to 0-100
        alignment_score = max(0, min(100, alignment_score * 100))
        
        # Analyze convergence trend
        convergence_trend = self._analyze_convergence_trend(
            agent_data, customer_data
        )
        
        result = {
            "call_id": call_id,
            "empathy_alignment_score": round(alignment_score, 2),
            "speech_rate_alignment": round(speech_rate_analysis["alignment_score"] * 100, 2),
            "pitch_alignment": round(pitch_analysis["alignment_score"] * 100, 2),
            "energy_alignment": round(energy_analysis["alignment_score"] * 100, 2),
            "convergence_trend": convergence_trend,
            "metrics": {
                "speech_rate": speech_rate_analysis,
                "pitch": pitch_analysis,
                "energy": energy_analysis,
            },
            "explainable_insights": self._generate_insights(
                speech_rate_analysis, pitch_analysis, energy_analysis, convergence_trend
            ),
        }
        
        logger.info(f"Empathy alignment score: {alignment_score:.2f}/100")
        
        return result
    
    def _extract_speaker_data(
        self,
        audio_features_df: pd.DataFrame,
        aligned_segments: List[Dict[str, any]],
        speaker_id: str,
    ) -> pd.DataFrame:
        """
        Extract time-ordered data for a speaker.
        
        Args:
            audio_features_df: DataFrame with audio features.
            aligned_segments: List of aligned segments with timestamps.
            speaker_id: Speaker identifier.
        
        Returns:
            DataFrame with speaker data sorted by time.
        """
        # Filter segments for this speaker
        speaker_segments = [
            seg for seg in aligned_segments if seg["speaker_id"] == speaker_id
        ]
        
        if not speaker_segments:
            return pd.DataFrame()
        
        # Create time-ordered data
        speaker_data = []
        for seg in sorted(speaker_segments, key=lambda x: x["start_time"]):
            # Find matching features
            seg_features = audio_features_df[
                (audio_features_df["speaker_id"] == speaker_id)
                & (audio_features_df["start_time"] <= seg["start_time"])
                & (audio_features_df["end_time"] >= seg["end_time"])
            ]
            
            if len(seg_features) > 0:
                row = seg_features.iloc[0].to_dict()
                row["segment_time"] = (seg["start_time"] + seg["end_time"]) / 2
                speaker_data.append(row)
        
        return pd.DataFrame(speaker_data)
    
    def _analyze_speech_rate_convergence(
        self,
        agent_data: pd.DataFrame,
        customer_data: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze speech rate convergence between agent and customer.
        
        Args:
            agent_data: Agent feature data over time.
            customer_data: Customer feature data over time.
        
        Returns:
            Dictionary with alignment score and metrics.
        """
        # Calculate speech rate from words per minute or segment duration
        # For now, use duration-based approximation
        agent_rates = self._calculate_speech_rates(agent_data)
        customer_rates = self._calculate_speech_rates(customer_data)
        
        if len(agent_rates) == 0 or len(customer_rates) == 0:
            return {"alignment_score": 0.5, "convergence": 0.0, "initial_diff": 0.0}
        
        # Calculate alignment (similarity)
        alignment = self._calculate_alignment(agent_rates, customer_rates)
        
        # Calculate convergence (trend towards similarity)
        convergence = self._calculate_convergence(agent_rates, customer_rates)
        
        # Initial difference
        initial_diff = abs(agent_rates[0] - customer_rates[0]) if len(agent_rates) > 0 and len(customer_rates) > 0 else 0.0
        
        # Combined score
        alignment_score = 0.6 * alignment + 0.4 * convergence
        
        return {
            "alignment_score": alignment_score,
            "alignment": alignment,
            "convergence": convergence,
            "initial_diff": initial_diff,
            "agent_mean_rate": float(np.mean(agent_rates)),
            "customer_mean_rate": float(np.mean(customer_rates)),
        }
    
    def _analyze_pitch_convergence(
        self,
        agent_data: pd.DataFrame,
        customer_data: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze pitch variation convergence between agent and customer.
        
        Args:
            agent_data: Agent feature data over time.
            customer_data: Customer feature data over time.
        
        Returns:
            Dictionary with alignment score and metrics.
        """
        agent_pitch = agent_data["pitch_mean"].values if "pitch_mean" in agent_data.columns else np.array([])
        customer_pitch = customer_data["pitch_mean"].values if "pitch_mean" in customer_data.columns else np.array([])
        
        if len(agent_pitch) == 0 or len(customer_pitch) == 0:
            return {"alignment_score": 0.5, "convergence": 0.0, "initial_diff": 0.0}
        
        # Calculate pitch variation (std)
        agent_pitch_std = np.std(agent_pitch)
        customer_pitch_std = np.std(customer_pitch)
        
        # Alignment: similarity in pitch variation
        alignment = 1.0 - abs(agent_pitch_std - customer_pitch_std) / max(agent_pitch_std, customer_pitch_std, 1.0)
        alignment = max(0.0, min(1.0, alignment))
        
        # Convergence: trend towards similar pitch
        convergence = self._calculate_convergence(agent_pitch, customer_pitch)
        
        # Initial difference
        initial_diff = abs(agent_pitch[0] - customer_pitch[0]) if len(agent_pitch) > 0 and len(customer_pitch) > 0 else 0.0
        
        # Combined score
        alignment_score = 0.6 * alignment + 0.4 * convergence
        
        return {
            "alignment_score": alignment_score,
            "alignment": alignment,
            "convergence": convergence,
            "initial_diff": initial_diff,
            "agent_pitch_mean": float(np.mean(agent_pitch)),
            "customer_pitch_mean": float(np.mean(customer_pitch)),
            "agent_pitch_std": float(agent_pitch_std),
            "customer_pitch_std": float(customer_pitch_std),
        }
    
    def _analyze_energy_convergence(
        self,
        agent_data: pd.DataFrame,
        customer_data: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze energy trend convergence between agent and customer.
        
        Args:
            agent_data: Agent feature data over time.
            customer_data: Customer feature data over time.
        
        Returns:
            Dictionary with alignment score and metrics.
        """
        agent_energy = agent_data["energy_mean"].values if "energy_mean" in agent_data.columns else np.array([])
        customer_energy = customer_data["energy_mean"].values if "energy_mean" in customer_data.columns else np.array([])
        
        if len(agent_energy) == 0 or len(customer_energy) == 0:
            return {"alignment_score": 0.5, "convergence": 0.0, "initial_diff": 0.0}
        
        # Calculate energy trends (slope)
        agent_trend = self._calculate_trend(agent_energy)
        customer_trend = self._calculate_trend(customer_energy)
        
        # Alignment: similarity in energy levels
        alignment = self._calculate_alignment(agent_energy, customer_energy)
        
        # Convergence: similar trends (both increasing/decreasing)
        trend_similarity = 1.0 - abs(agent_trend - customer_trend) / max(abs(agent_trend), abs(customer_trend), 0.01)
        trend_similarity = max(0.0, min(1.0, trend_similarity))
        
        # Combined convergence
        convergence = 0.5 * trend_similarity + 0.5 * self._calculate_convergence(agent_energy, customer_energy)
        
        # Combined score
        alignment_score = 0.6 * alignment + 0.4 * convergence
        
        return {
            "alignment_score": alignment_score,
            "alignment": alignment,
            "convergence": convergence,
            "agent_energy_trend": float(agent_trend),
            "customer_energy_trend": float(customer_trend),
            "agent_energy_mean": float(np.mean(agent_energy)),
            "customer_energy_mean": float(np.mean(customer_energy)),
        }
    
    def _calculate_speech_rates(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate speech rates from data."""
        if "duration" in data.columns and len(data) > 0:
            # Approximate: shorter duration = faster speech (inverse relationship)
            durations = data["duration"].values
            # Normalize and invert
            rates = 1.0 / (durations + 0.1)  # Add small value to avoid division by zero
            return rates
        return np.array([])
    
    def _calculate_alignment(
        self,
        agent_values: np.ndarray,
        customer_values: np.ndarray,
    ) -> float:
        """
        Calculate alignment (similarity) between agent and customer values.
        
        Args:
            agent_values: Agent feature values.
            customer_values: Customer feature values.
        
        Returns:
            Alignment score (0-1).
        """
        if len(agent_values) == 0 or len(customer_values) == 0:
            return 0.5
        
        # Interpolate to same length
        min_len = min(len(agent_values), len(customer_values))
        agent_aligned = agent_values[:min_len]
        customer_aligned = customer_values[:min_len]
        
        # Calculate correlation
        if len(agent_aligned) > 1:
            correlation = np.corrcoef(agent_aligned, customer_aligned)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Calculate mean difference
        mean_diff = abs(np.mean(agent_aligned) - np.mean(customer_aligned))
        max_val = max(np.max(agent_aligned), np.max(customer_aligned), 1.0)
        similarity = 1.0 - (mean_diff / max_val)
        similarity = max(0.0, min(1.0, similarity))
        
        # Combine correlation and similarity
        alignment = 0.5 * (correlation + 1.0) + 0.5 * similarity
        
        return max(0.0, min(1.0, alignment))
    
    def _calculate_convergence(
        self,
        agent_values: np.ndarray,
        customer_values: np.ndarray,
    ) -> float:
        """
        Calculate convergence (trend towards similarity) over time.
        
        Args:
            agent_values: Agent feature values over time.
            customer_values: Customer feature values over time.
        
        Returns:
            Convergence score (0-1).
        """
        if len(agent_values) < 2 or len(customer_values) < 2:
            return 0.5
        
        # Interpolate to same length
        min_len = min(len(agent_values), len(customer_values))
        agent_aligned = agent_values[:min_len]
        customer_aligned = customer_values[:min_len]
        
        # Calculate difference over time
        differences = np.abs(agent_aligned - customer_aligned)
        
        # Calculate trend (negative slope = convergence)
        if len(differences) > 1:
            x = np.arange(len(differences))
            slope, _, _, _, _ = stats.linregress(x, differences)
            
            # Normalize slope to [0, 1] range
            # Negative slope = convergence (good)
            max_slope = np.max(np.abs(differences)) if len(differences) > 0 else 1.0
            convergence = 1.0 - (slope / max_slope) if max_slope > 0 else 0.5
            convergence = max(0.0, min(1.0, convergence))
        else:
            convergence = 0.5
        
        return convergence
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """
        Calculate trend (slope) of values over time.
        
        Args:
            values: Feature values over time.
        
        Returns:
            Trend slope.
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        return float(slope)
    
    def _analyze_convergence_trend(
        self,
        agent_data: pd.DataFrame,
        customer_data: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Analyze overall convergence trend over time.
        
        Args:
            agent_data: Agent feature data.
            customer_data: Customer feature data.
        
        Returns:
            Dictionary with convergence trend analysis.
        """
        # Combine all metrics for overall convergence
        speech_rate_conv = self._analyze_speech_rate_convergence(agent_data, customer_data)["convergence"]
        pitch_conv = self._analyze_pitch_convergence(agent_data, customer_data)["convergence"]
        energy_conv = self._analyze_energy_convergence(agent_data, customer_data)["convergence"]
        
        overall_convergence = (
            self.speech_rate_weight * speech_rate_conv
            + self.pitch_weight * pitch_conv
            + self.energy_weight * energy_conv
        )
        
        return {
            "overall_convergence": float(overall_convergence),
            "speech_rate_convergence": float(speech_rate_conv),
            "pitch_convergence": float(pitch_conv),
            "energy_convergence": float(energy_conv),
            "trend": "converging" if overall_convergence > 0.6 else "diverging" if overall_convergence < 0.4 else "stable",
        }
    
    def _generate_insights(
        self,
        speech_rate_analysis: Dict[str, any],
        pitch_analysis: Dict[str, any],
        energy_analysis: Dict[str, any],
        convergence_trend: Dict[str, any],
    ) -> List[str]:
        """
        Generate explainable insights from analysis.
        
        Args:
            speech_rate_analysis: Speech rate analysis results.
            pitch_analysis: Pitch analysis results.
            energy_analysis: Energy analysis results.
            convergence_trend: Convergence trend analysis.
        
        Returns:
            List of insight strings.
        """
        insights = []
        
        # Overall score insight
        overall_score = (
            self.speech_rate_weight * speech_rate_analysis["alignment_score"]
            + self.pitch_weight * pitch_analysis["alignment_score"]
            + self.energy_weight * energy_analysis["alignment_score"]
        ) * 100
        
        if overall_score >= 75:
            insights.append("High empathy alignment: Agent and customer show strong communication synchronization.")
        elif overall_score >= 50:
            insights.append("Moderate empathy alignment: Some synchronization present but room for improvement.")
        else:
            insights.append("Low empathy alignment: Limited communication synchronization detected.")
        
        # Convergence trend
        if convergence_trend["trend"] == "converging":
            insights.append("Positive trend: Communication patterns are converging over time.")
        elif convergence_trend["trend"] == "diverging":
            insights.append("Negative trend: Communication patterns are diverging over time.")
        else:
            insights.append("Stable pattern: Communication patterns remain relatively constant.")
        
        # Speech rate insight
        sr_diff = abs(speech_rate_analysis.get("agent_mean_rate", 0) - speech_rate_analysis.get("customer_mean_rate", 0))
        if sr_diff < 0.1:
            insights.append("Speech rate: Agent and customer speak at similar pace.")
        else:
            insights.append(f"Speech rate: Noticeable difference in speaking pace (diff: {sr_diff:.2f}).")
        
        # Pitch insight
        pitch_align = pitch_analysis.get("alignment", 0.5)
        if pitch_align > 0.7:
            insights.append("Pitch: Similar pitch variation patterns indicate good vocal alignment.")
        else:
            insights.append("Pitch: Different pitch variation patterns suggest limited vocal synchronization.")
        
        # Energy insight
        energy_align = energy_analysis.get("alignment", 0.5)
        if energy_align > 0.7:
            insights.append("Energy: Similar energy levels indicate good engagement alignment.")
        else:
            insights.append("Energy: Different energy levels suggest engagement mismatch.")
        
        return insights
    
    def _empty_result(self, call_id: str) -> Dict[str, any]:
        """Return empty result for insufficient data."""
        return {
            "call_id": call_id,
            "empathy_alignment_score": 0.0,
            "speech_rate_alignment": 0.0,
            "pitch_alignment": 0.0,
            "energy_alignment": 0.0,
            "convergence_trend": {"trend": "insufficient_data"},
            "metrics": {},
            "explainable_insights": ["Insufficient data for analysis."],
        }


def analyze_empathy_convergence(
    call_id: str,
    audio_features_path: str | Path,
    diarization_path: str | Path,
    output_path: Optional[str | Path] = None,
    agent_speaker_id: str = "SPEAKER_00",
    customer_speaker_id: str = "SPEAKER_01",
) -> Dict[str, any]:
    """
    Convenience function to analyze empathy convergence for a call.
    
    Args:
        call_id: Call identifier.
        audio_features_path: Path to audio features CSV file.
        diarization_path: Path to diarization JSON file.
        output_path: Optional path to save JSON results.
        agent_speaker_id: Speaker ID for agent. Default is "SPEAKER_00".
        customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
    
    Returns:
        Dictionary containing empathy convergence analysis results.
    
    Example:
        >>> result = analyze_empathy_convergence(
        ...     call_id="call_abc123",
        ...     audio_features_path="data/processed/audio_features_call_abc123.csv",
        ...     diarization_path="data/processed/transcripts/call_abc123/diarization.json"
        ... )
    """
    analyzer = EmpathyConvergenceAnalyzer()
    
    result = analyzer.analyze_call(
        call_id=call_id,
        audio_features_path=audio_features_path,
        diarization_path=diarization_path,
        agent_speaker_id=agent_speaker_id,
        customer_speaker_id=customer_speaker_id,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved empathy convergence analysis to: {output_path}")
    
    return result
