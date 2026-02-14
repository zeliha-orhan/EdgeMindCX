"""
Speaker-aware speech dynamics analysis module for EdgeMindCX project.

Analyzes speech patterns, silence, and overlaps for each speaker in call center audio.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SpeakerDynamicsAnalyzer:
    """
    Analyzes speaker-aware speech dynamics in call center conversations.
    
    Features:
    - Speech rate (words per minute) per speaker
    - Silence detection per speaker
    - Agent vs Customer analysis
    - Overlap detection (simultaneous speech)
    """
    
    def __init__(
        self,
        agent_speaker_id: str = "SPEAKER_00",
        customer_speaker_id: str = "SPEAKER_01",
        silence_threshold: float = 0.5,
    ) -> None:
        """
        Initialize speaker dynamics analyzer.
        
        Args:
            agent_speaker_id: Speaker ID for agent. Default is "SPEAKER_00".
            customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
            silence_threshold: Minimum silence duration in seconds to be counted.
                              Default is 0.5 seconds.
        """
        self.agent_speaker_id = agent_speaker_id
        self.customer_speaker_id = customer_speaker_id
        self.silence_threshold = silence_threshold
    
    def analyze_call(
        self,
        call_id: str,
        diarization_data: Dict[str, any],
        transcription_data: Optional[Dict[str, any]] = None,
    ) -> List[Dict[str, any]]:
        """
        Analyze speech dynamics for a call.
        
        Args:
            call_id: Call identifier.
            diarization_data: Diarization result dictionary containing aligned_segments.
            transcription_data: Optional transcription data for word-level analysis.
        
        Returns:
            List of metric dictionaries, each containing:
            - call_id: Call identifier
            - speaker: Speaker ID or role (agent/customer)
            - metric: Metric name
            - value: Metric value
        """
        logger.info(f"Analyzing speech dynamics for call: {call_id}")
        
        aligned_segments = diarization_data.get("aligned_segments", [])
        diarization_segments = diarization_data.get("diarization_segments", [])
        
        if not aligned_segments:
            logger.warning(f"No aligned segments found for call {call_id}")
            return []
        
        metrics = []
        
        # Get all unique speakers
        speakers = set(seg["speaker_id"] for seg in aligned_segments)
        
        # Analyze each speaker
        for speaker_id in speakers:
            speaker_segments = [
                seg for seg in aligned_segments if seg["speaker_id"] == speaker_id
            ]
            
            # Determine role
            role = self._determine_role(speaker_id)
            
            # Calculate metrics
            speaker_metrics = self._calculate_speaker_metrics(
                call_id=call_id,
                speaker_id=speaker_id,
                role=role,
                segments=speaker_segments,
                all_segments=aligned_segments,
                diarization_segments=diarization_segments,
            )
            
            metrics.extend(speaker_metrics)
        
        # Calculate overlap metrics
        overlap_metrics = self._calculate_overlap_metrics(
            call_id=call_id,
            aligned_segments=aligned_segments,
        )
        metrics.extend(overlap_metrics)
        
        logger.info(f"Calculated {len(metrics)} metrics for call {call_id}")
        
        return metrics
    
    def _determine_role(self, speaker_id: str) -> str:
        """
        Determine speaker role (agent or customer).
        
        Args:
            speaker_id: Speaker identifier.
        
        Returns:
            Role string: "agent", "customer", or "unknown".
        """
        if speaker_id == self.agent_speaker_id:
            return "agent"
        elif speaker_id == self.customer_speaker_id:
            return "customer"
        else:
            return "unknown"
    
    def _calculate_speaker_metrics(
        self,
        call_id: str,
        speaker_id: str,
        role: str,
        segments: List[Dict[str, any]],
        all_segments: List[Dict[str, any]],
        diarization_segments: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Calculate all metrics for a single speaker.
        
        Args:
            call_id: Call identifier.
            speaker_id: Speaker identifier.
            role: Speaker role (agent/customer/unknown).
            segments: Segments for this speaker.
            all_segments: All segments in the call.
            diarization_segments: Raw diarization segments.
        
        Returns:
            List of metric dictionaries.
        """
        metrics = []
        
        # Speech rate (words per minute)
        wpm = self._calculate_speech_rate(segments)
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "words_per_minute",
            "value": wpm,
        })
        
        # Total speaking time
        speaking_time = sum(
            seg["end_time"] - seg["start_time"] for seg in segments
        )
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "total_speaking_time_seconds",
            "value": speaking_time,
        })
        
        # Number of segments
        num_segments = len(segments)
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "num_segments",
            "value": num_segments,
        })
        
        # Average segment duration
        if num_segments > 0:
            avg_segment_duration = speaking_time / num_segments
            metrics.append({
                "call_id": call_id,
                "speaker": role,
                "metric": "avg_segment_duration_seconds",
                "value": avg_segment_duration,
            })
        
        # Silence analysis
        silence_metrics = self._calculate_silence_metrics(
            call_id=call_id,
            speaker_id=speaker_id,
            role=role,
            speaker_segments=segments,
            all_segments=all_segments,
            diarization_segments=diarization_segments,
        )
        metrics.extend(silence_metrics)
        
        # Word count
        total_words = sum(
            len(seg.get("text", "").split()) for seg in segments
        )
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "total_words",
            "value": total_words,
        })
        
        return metrics
    
    def _calculate_speech_rate(
        self,
        segments: List[Dict[str, any]],
    ) -> float:
        """
        Calculate speech rate in words per minute (WPM).
        
        Args:
            segments: List of speaker segments.
        
        Returns:
            Words per minute (float).
        """
        if not segments:
            return 0.0
        
        # Count total words
        total_words = sum(
            len(seg.get("text", "").split()) for seg in segments
        )
        
        # Calculate total speaking time
        total_time = sum(
            seg["end_time"] - seg["start_time"] for seg in segments
        )
        
        if total_time == 0:
            return 0.0
        
        # Calculate WPM
        wpm = (total_words / total_time) * 60.0
        
        return round(wpm, 2)
    
    def _calculate_silence_metrics(
        self,
        call_id: str,
        speaker_id: str,
        role: str,
        speaker_segments: List[Dict[str, any]],
        all_segments: List[Dict[str, any]],
        diarization_segments: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Calculate silence-related metrics for a speaker.
        
        Args:
            call_id: Call identifier.
            speaker_id: Speaker identifier.
            role: Speaker role.
            speaker_segments: Segments for this speaker.
            all_segments: All segments in the call.
            diarization_segments: Raw diarization segments.
        
        Returns:
            List of silence metric dictionaries.
        """
        metrics = []
        
        # Find silence periods for this speaker
        # Silence = gaps between this speaker's segments
        speaker_segments_sorted = sorted(
            speaker_segments, key=lambda x: x["start_time"]
        )
        
        silence_periods = []
        for i in range(len(speaker_segments_sorted) - 1):
            current_end = speaker_segments_sorted[i]["end_time"]
            next_start = speaker_segments_sorted[i + 1]["start_time"]
            gap = next_start - current_end
            
            if gap >= self.silence_threshold:
                silence_periods.append({
                    "start": current_end,
                    "end": next_start,
                    "duration": gap,
                })
        
        # Total silence time for this speaker
        total_silence = sum(period["duration"] for period in silence_periods)
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "total_silence_seconds",
            "value": round(total_silence, 2),
        })
        
        # Number of silence periods
        num_silence_periods = len(silence_periods)
        metrics.append({
            "call_id": call_id,
            "speaker": role,
            "metric": "num_silence_periods",
            "value": num_silence_periods,
        })
        
        # Average silence duration
        if num_silence_periods > 0:
            avg_silence = total_silence / num_silence_periods
            metrics.append({
                "call_id": call_id,
                "speaker": role,
                "metric": "avg_silence_duration_seconds",
                "value": round(avg_silence, 2),
            })
        
        # Longest silence period
        if silence_periods:
            longest_silence = max(period["duration"] for period in silence_periods)
            metrics.append({
                "call_id": call_id,
                "speaker": role,
                "metric": "longest_silence_seconds",
                "value": round(longest_silence, 2),
            })
        
        return metrics
    
    def _calculate_overlap_metrics(
        self,
        call_id: str,
        aligned_segments: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Calculate overlap (simultaneous speech) metrics.
        
        Args:
            call_id: Call identifier.
            aligned_segments: All aligned segments.
        
        Returns:
            List of overlap metric dictionaries.
        """
        metrics = []
        
        # Find overlapping segments
        overlaps = []
        segments_sorted = sorted(
            aligned_segments, key=lambda x: x["start_time"]
        )
        
        for i in range(len(segments_sorted)):
            for j in range(i + 1, len(segments_sorted)):
                seg1 = segments_sorted[i]
                seg2 = segments_sorted[j]
                
                # Check if segments overlap
                if seg1["speaker_id"] != seg2["speaker_id"]:
                    overlap_start = max(seg1["start_time"], seg2["start_time"])
                    overlap_end = min(seg1["end_time"], seg2["end_time"])
                    
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        overlaps.append({
                            "speaker1": seg1["speaker_id"],
                            "speaker2": seg2["speaker_id"],
                            "start": overlap_start,
                            "end": overlap_end,
                            "duration": overlap_duration,
                        })
        
        # Total overlap time
        total_overlap = sum(overlap["duration"] for overlap in overlaps)
        metrics.append({
            "call_id": call_id,
            "speaker": "overlap",
            "metric": "total_overlap_seconds",
            "value": round(total_overlap, 2),
        })
        
        # Number of overlaps
        num_overlaps = len(overlaps)
        metrics.append({
            "call_id": call_id,
            "speaker": "overlap",
            "metric": "num_overlaps",
            "value": num_overlaps,
        })
        
        # Average overlap duration
        if num_overlaps > 0:
            avg_overlap = total_overlap / num_overlaps
            metrics.append({
                "call_id": call_id,
                "speaker": "overlap",
                "metric": "avg_overlap_duration_seconds",
                "value": round(avg_overlap, 2),
            })
        
        # Longest overlap
        if overlaps:
            longest_overlap = max(overlap["duration"] for overlap in overlaps)
            metrics.append({
                "call_id": call_id,
                "speaker": "overlap",
                "metric": "longest_overlap_seconds",
                "value": round(longest_overlap, 2),
            })
        
        return metrics
    
    def save_to_csv(
        self,
        metrics: List[Dict[str, any]],
        output_path: str | Path,
    ) -> Path:
        """
        Save metrics to CSV file.
        
        Args:
            metrics: List of metric dictionaries.
            output_path: Path to output CSV file.
        
        Returns:
            Path to saved CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(metrics)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved metrics to CSV: {output_path}")
        
        return output_path


def analyze_speaker_dynamics(
    call_id: str,
    diarization_path: str | Path,
    transcription_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    agent_speaker_id: str = "SPEAKER_00",
    customer_speaker_id: str = "SPEAKER_01",
) -> List[Dict[str, any]]:
    """
    Convenience function to analyze speaker dynamics for a call.
    
    Args:
        call_id: Call identifier.
        diarization_path: Path to diarization JSON file.
        transcription_path: Optional path to transcription JSON file.
        output_path: Optional path to save CSV output.
        agent_speaker_id: Speaker ID for agent. Default is "SPEAKER_00".
        customer_speaker_id: Speaker ID for customer. Default is "SPEAKER_01".
    
    Returns:
        List of metric dictionaries.
    
    Example:
        >>> metrics = analyze_speaker_dynamics(
        ...     call_id="call_abc123",
        ...     diarization_path="data/processed/transcripts/call_abc123/diarization.json"
        ... )
    """
    # Load diarization data
    diarization_path = Path(diarization_path)
    with open(diarization_path, "r", encoding="utf-8") as f:
        diarization_data = json.load(f)
    
    # Load transcription data if provided
    transcription_data = None
    if transcription_path:
        transcription_path = Path(transcription_path)
        if transcription_path.exists():
            with open(transcription_path, "r", encoding="utf-8") as f:
                transcription_data = json.load(f)
    
    # Analyze
    analyzer = SpeakerDynamicsAnalyzer(
        agent_speaker_id=agent_speaker_id,
        customer_speaker_id=customer_speaker_id,
    )
    
    metrics = analyzer.analyze_call(
        call_id=call_id,
        diarization_data=diarization_data,
        transcription_data=transcription_data,
    )
    
    # Save to CSV if output path provided
    if output_path:
        analyzer.save_to_csv(metrics, output_path)
    
    return metrics
