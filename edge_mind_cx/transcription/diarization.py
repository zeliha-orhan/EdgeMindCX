"""
Speaker diarization module for EdgeMindCX project.

Uses pyannote.audio to identify and segment speakers in call center audio.
Aligns speaker segments with transcription timestamps.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import librosa
    import torch
    from pyannote.audio import Pipeline
except ImportError:
    librosa = None
    torch = None
    Pipeline = None
    logging.warning(
        "pyannote.audio not installed. Install with: pip install pyannote.audio"
    )

logger = logging.getLogger(__name__)

# Global cache for diarization pipelines (singleton pattern)
_diarization_pipeline_cache: Dict[str, any] = {}


class SpeakerDiarization:
    """
    Speaker diarization using pyannote.audio.
    
    Identifies speakers in audio and segments them by time.
    Aligns speaker segments with transcription timestamps.
    """
    
    def __init__(
        self,
        min_speakers: int = 2,
        max_speakers: Optional[int] = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: Optional[str] = None,
        use_simple_heuristic_only: bool = False,
    ) -> None:
        """
        Initialize speaker diarization.
        
        Args:
            min_speakers: Minimum number of speakers. Default is 2.
            max_speakers: Maximum number of speakers. If None, auto-detects.
            model_name: pyannote.audio model name. Default is speaker-diarization-3.1.
            use_auth_token: HuggingFace auth token if model requires it.
            device: Device to run model on ('cpu' or 'cuda'). If None, auto-detects.
            use_simple_heuristic_only: If True, skip pyannote initialization (for fast path).
        
        Raises:
            ImportError: If pyannote.audio is not installed and use_simple_heuristic_only is False.
        """
        self.use_simple_heuristic_only = use_simple_heuristic_only
        
        if use_simple_heuristic_only:
            # Skip pyannote initialization for fast path
            logger.info("Using simple heuristic only - skipping pyannote.audio initialization")
            self.pipeline = None
            self.min_speakers = min_speakers
            self.max_speakers = max_speakers
            self.model_name = model_name
            self.use_auth_token = use_auth_token
            self.device = device
            return
        
        if Pipeline is None:
            raise ImportError(
                "pyannote.audio is required. Install with: pip install pyannote.audio"
            )
        
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.model_name = model_name
        self.use_auth_token = use_auth_token
        self.device = device
        
        # Initialize pyannote pipeline (use global cache to avoid reloading)
        cache_key = f"{model_name}_{device or 'auto'}"
        
        if cache_key in _diarization_pipeline_cache:
            logger.info(f"Using cached pyannote.audio pipeline: {model_name}")
            self.pipeline = _diarization_pipeline_cache[cache_key]
        else:
            logger.info(f"Loading pyannote.audio model: {model_name} (first time, will be cached)")
            try:
                # New pyannote.audio versions use 'token' instead of 'use_auth_token'
                pipeline_kwargs = {}
                if use_auth_token:
                    pipeline_kwargs["token"] = use_auth_token
                    logger.info("Using HuggingFace token for model access")
                else:
                    logger.warning("No HuggingFace token provided. Model access may fail if gated.")
                
                self.pipeline = Pipeline.from_pretrained(
                    model_name,
                    **pipeline_kwargs,
                )
                
                if device:
                    self.pipeline.to(torch.device(device))
                
                _diarization_pipeline_cache[cache_key] = self.pipeline
                logger.info("pyannote.audio model loaded and cached successfully")
            except Exception as e:
                logger.error(f"Error loading pyannote.audio model: {e}")
                raise
    
    def _simple_2speaker_heuristic(
        self,
        audio_path: str | Path,
        transcription_segments: Optional[List[Dict[str, any]]] = None,
    ) -> List[Dict[str, any]]:
        """
        Simple 2-speaker heuristic for short audio (< 30s).
        
        Alternates speakers based on transcription segments or time-based splitting.
        This is a fast alternative to full pyannote.audio diarization.
        
        Args:
            audio_path: Path to audio file.
            transcription_segments: Optional transcription segments for better alignment.
        
        Returns:
            List of speaker segments with alternating speakers.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Using simple 2-speaker heuristic for: {audio_path.name}")
        
        # Get audio duration
        try:
            import soundfile as sf
            audio_info = sf.info(str(audio_path))
            duration = audio_info.duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}, using transcription segments")
            duration = None
        
        segments = []
        
        if transcription_segments and len(transcription_segments) > 0:
            # Use transcription segments to alternate speakers
            current_speaker = "SPEAKER_00"
            for i, seg in enumerate(transcription_segments):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                
                # Alternate speakers every segment
                if i > 0 and i % 2 == 0:
                    current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                
                segments.append({
                    "speaker_id": current_speaker,
                    "start_time": start,
                    "end_time": end,
                })
        else:
            # Time-based splitting: alternate every 5 seconds
            if duration is None:
                duration = 30.0  # Default fallback
            
            chunk_duration = max(2.0, duration / 4)  # At least 2s chunks, max 4 chunks
            current_speaker = "SPEAKER_00"
            
            current_time = 0.0
            while current_time < duration:
                end_time = min(current_time + chunk_duration, duration)
                
                segments.append({
                    "speaker_id": current_speaker,
                    "start_time": current_time,
                    "end_time": end_time,
                })
                
                # Alternate speaker
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                current_time = end_time
        
        logger.info(
            f"Simple heuristic completed: {len(segments)} segments, "
            f"{len(set(s['speaker_id'] for s in segments))} speakers"
        )
        
        return segments
    
    def diarize_audio(
        self,
        audio_path: str | Path,
        use_simple_heuristic: bool = False,
        transcription_segments: Optional[List[Dict[str, any]]] = None,
    ) -> List[Dict[str, any]]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file.
            use_simple_heuristic: If True, use simple 2-speaker heuristic instead of pyannote.
            transcription_segments: Optional transcription segments for heuristic alignment.
        
        Returns:
            List of speaker segments, each containing:
            - speaker_id: Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01")
            - start_time: Start time in seconds
            - end_time: End time in seconds
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use simple heuristic if requested (for fast path)
        if use_simple_heuristic:
            return self._simple_2speaker_heuristic(audio_path, transcription_segments)
        
        # Check if pipeline is available
        if self.pipeline is None:
            raise RuntimeError(
                "pyannote.audio pipeline not initialized. "
                "Either initialize with use_simple_heuristic_only=False or use simple heuristic."
            )
        
        logger.info(f"Running speaker diarization on: {audio_path.name}")
        
        try:
            # Load audio using librosa (to avoid torchcodec/FFmpeg dependency)
            logger.debug(f"Loading audio file: {audio_path}")
            audio_array, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
            logger.debug(f"Audio loaded: {len(audio_array)} samples, {sample_rate}Hz")
            
            # Convert to torch tensor and prepare for pyannote
            # pyannote expects shape: (channels, time)
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # (1, time)
            
            # Prepare audio dictionary for pyannote
            audio_dict = {
                "waveform": audio_tensor,
                "sample_rate": sample_rate,
            }
            
            # Run diarization
            diarization_result = self.pipeline(
                audio_dict,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            
            # Extract segments
            # New pyannote.audio 4.0+ API: DiarizeOutput has speaker_diarization attribute
            segments = []
            try:
                # Access the annotation object from DiarizeOutput
                annotation = diarization_result.speaker_diarization
                for segment, track, label in annotation.itertracks(yield_label=True):
                    segments.append({
                        "speaker_id": label,
                        "start_time": segment.start,
                        "end_time": segment.end,
                    })
            except AttributeError:
                # Fallback: try itertracks directly if it's an Annotation
                try:
                    for segment, track, label in diarization_result.itertracks(yield_label=True):
                        segments.append({
                            "speaker_id": label,
                            "start_time": segment.start,
                            "end_time": segment.end,
                        })
                except AttributeError:
                    logger.error(f"Unexpected diarization result type: {type(diarization_result)}")
                    logger.error(f"Available attributes: {[x for x in dir(diarization_result) if not x.startswith('_')]}")
                    raise
            
            # Sort by start time
            segments.sort(key=lambda x: x["start_time"])
            
            logger.info(
                f"Diarization completed: {len(segments)} segments, "
                f"{len(set(s['speaker_id'] for s in segments))} speakers detected"
            )
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            raise
    
    def align_with_transcription(
        self,
        diarization_segments: List[Dict[str, any]],
        transcription_segments: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Align speaker diarization segments with transcription segments.
        
        For each transcription segment, finds the corresponding speaker
        based on time overlap.
        
        Args:
            diarization_segments: List of speaker diarization segments.
            transcription_segments: List of transcription segments with timestamps.
        
        Returns:
            List of aligned segments, each containing:
            - speaker_id: Speaker identifier
            - start_time: Start time in seconds
            - end_time: End time in seconds
            - text: Transcription text
            - segment_id: Original transcription segment ID
        """
        logger.debug("Aligning diarization with transcription...")
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get("start", 0.0)
            trans_end = trans_seg.get("end", 0.0)
            trans_text = trans_seg.get("text", "").strip()
            trans_id = trans_seg.get("id")
            
            # Find overlapping speaker segment
            speaker_id = self._find_speaker_for_time(
                diarization_segments,
                trans_start,
                trans_end,
            )
            
            aligned_segments.append({
                "speaker_id": speaker_id,
                "start_time": trans_start,
                "end_time": trans_end,
                "text": trans_text,
                "segment_id": trans_id,
            })
        
        logger.debug(f"Aligned {len(aligned_segments)} segments")
        
        return aligned_segments
    
    def _find_speaker_for_time(
        self,
        diarization_segments: List[Dict[str, any]],
        start_time: float,
        end_time: float,
    ) -> str:
        """
        Find speaker ID for a given time range.
        
        Uses the speaker segment that has the most overlap with the time range.
        
        Args:
            diarization_segments: List of speaker diarization segments.
            start_time: Start time of the segment.
            end_time: End time of the segment.
        
        Returns:
            Speaker ID string.
        """
        best_speaker = None
        max_overlap = 0.0
        
        for diar_seg in diarization_segments:
            diar_start = diar_seg["start_time"]
            diar_end = diar_seg["end_time"]
            
            # Calculate overlap
            overlap_start = max(start_time, diar_start)
            overlap_end = min(end_time, diar_end)
            overlap = max(0.0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg["speaker_id"]
        
        # If no overlap found, use the nearest speaker segment
        if best_speaker is None and diarization_segments:
            # Find nearest segment
            nearest_seg = min(
                diarization_segments,
                key=lambda x: min(
                    abs(x["start_time"] - start_time),
                    abs(x["end_time"] - end_time),
                ),
            )
            best_speaker = nearest_seg["speaker_id"]
        
        return best_speaker or "UNKNOWN"
    
    def process_call(
        self,
        audio_path: str | Path,
        transcription_path: Optional[str | Path] = None,
        call_id: Optional[str] = None,
        use_simple_heuristic: bool = False,
    ) -> Dict[str, any]:
        """
        Process a call: perform diarization and align with transcription.
        
        Args:
            audio_path: Path to audio file.
            transcription_path: Path to transcription JSON file.
                              If None, will look for transcript.json in call_id directory.
            call_id: Call ID. Used to locate transcription if transcription_path is None.
            use_simple_heuristic: If True, use simple 2-speaker heuristic (fast path).
        
        Returns:
            Dictionary containing:
            - call_id: Call identifier
            - audio_path: Path to audio file
            - diarization_segments: Raw diarization segments
            - aligned_segments: Segments aligned with transcription
            - speakers: List of unique speaker IDs
            - is_fast_path: Whether simple heuristic was used
        """
        audio_path = Path(audio_path)
        
        # Load transcription if provided
        transcription_segments = []
        if transcription_path:
            transcription_path = Path(transcription_path)
            if transcription_path.exists():
                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcription_data = json.load(f)
                    transcription_segments = transcription_data.get("segments", [])
        elif call_id:
            # Try to find segments.json in call_id directory (preferred) or transcript.json
            segments_path = Path("data/processed/transcripts") / call_id / "segments.json"
            if segments_path.exists():
                with open(segments_path, "r", encoding="utf-8") as f:
                    transcription_data = json.load(f)
                    transcription_segments = transcription_data.get("segments", [])
            else:
                # Fallback to transcript.json
                transcript_path = Path("data/processed/transcripts") / call_id / "transcript.json"
                if transcript_path.exists():
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcription_data = json.load(f)
                        transcription_segments = transcription_data.get("segments", [])
        
        # Perform diarization (with or without simple heuristic)
        diarization_segments = self.diarize_audio(
            audio_path,
            use_simple_heuristic=use_simple_heuristic,
            transcription_segments=transcription_segments if use_simple_heuristic else None,
        )
        
        # Align with transcription if available
        aligned_segments = []
        if transcription_segments:
            aligned_segments = self.align_with_transcription(
                diarization_segments,
                transcription_segments,
            )
        else:
            logger.warning("No transcription found, returning raw diarization segments")
            aligned_segments = diarization_segments
        
        # Extract unique speakers
        speakers = sorted(list(set(s["speaker_id"] for s in diarization_segments)))
        
        # Calculate speaking durations for each speaker
        speaker_durations = self._calculate_speaker_durations(
            diarization_segments,
            aligned_segments,
        )
        
        result = {
            "call_id": call_id or audio_path.stem,
            "audio_path": str(audio_path),
            "diarization_segments": diarization_segments,
            "aligned_segments": aligned_segments,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "speaker_durations": speaker_durations,
            "is_fast_path": use_simple_heuristic,
        }
        
        logger.info(
            f"Diarization completed for {result['call_id']}: "
            f"{len(diarization_segments)} segments, {len(speakers)} speakers"
        )
        
        return result
    
    def _calculate_speaker_durations(
        self,
        diarization_segments: List[Dict[str, any]],
        aligned_segments: List[Dict[str, any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate speaking durations for each speaker.
        
        Args:
            diarization_segments: Raw diarization segments.
            aligned_segments: Aligned segments with transcription.
        
        Returns:
            Dictionary with speaker durations:
            - {speaker_id: {"total_duration": float, "num_segments": int}}
        """
        durations = {}
        
        # Calculate from diarization segments (more accurate for raw durations)
        for seg in diarization_segments:
            speaker_id = seg["speaker_id"]
            duration = seg["end_time"] - seg["start_time"]
            
            if speaker_id not in durations:
                durations[speaker_id] = {
                    "total_duration": 0.0,
                    "num_segments": 0,
                }
            
            durations[speaker_id]["total_duration"] += duration
            durations[speaker_id]["num_segments"] += 1
        
        # Also calculate from aligned segments if available (for text-based analysis)
        if aligned_segments:
            aligned_durations = {}
            for seg in aligned_segments:
                speaker_id = seg.get("speaker_id", "UNKNOWN")
                duration = seg.get("end_time", 0.0) - seg.get("start_time", 0.0)
                
                if speaker_id not in aligned_durations:
                    aligned_durations[speaker_id] = {
                        "total_duration": 0.0,
                        "num_segments": 0,
                    }
                
                aligned_durations[speaker_id]["total_duration"] += duration
                aligned_durations[speaker_id]["num_segments"] += 1
            
            # Add aligned durations info to result
            for speaker_id, info in durations.items():
                if speaker_id in aligned_durations:
                    info["aligned_duration"] = aligned_durations[speaker_id]["total_duration"]
                    info["aligned_segments"] = aligned_durations[speaker_id]["num_segments"]
        
        return durations
    
    def save_diarization(
        self,
        diarization_result: Dict[str, any],
        output_dir: Optional[str | Path] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save diarization results to JSON file.
        
        Args:
            diarization_result: Diarization result dictionary.
            output_dir: Output directory. If None, uses data/processed/diarization.
            filename: Output filename. If None, uses {call_id}.json.
        
        Returns:
            Path to saved JSON file.
        """
        call_id = diarization_result["call_id"]
        
        if output_dir is None:
            output_dir = Path("data/processed/diarization")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{call_id}.json"
        
        output_file = output_dir / filename
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(diarization_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved diarization results: {output_file}")
        
        return output_file


def diarize_call(
    audio_path: str | Path,
    transcription_path: Optional[str | Path] = None,
    call_id: Optional[str] = None,
    min_speakers: int = 2,
    max_speakers: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    save_results: bool = True,
    use_simple_heuristic: bool = False,
) -> Dict[str, any]:
    """
    Convenience function to perform speaker diarization on a call.
    
    Args:
        audio_path: Path to audio file.
        transcription_path: Path to transcription JSON file.
        call_id: Call ID.
        min_speakers: Minimum number of speakers. Default is 2.
        max_speakers: Maximum number of speakers. If None, auto-detects.
        output_dir: Output directory for saving results.
        save_results: Whether to save results to JSON. Default is True.
        use_simple_heuristic: If True, use simple 2-speaker heuristic (fast path).
    
    Returns:
        Dictionary containing diarization results.
    
    Example:
        >>> result = diarize_call(
        ...     audio_path="data/raw/audio/call_center/call1.wav",
        ...     call_id="call_abc123",
        ...     use_simple_heuristic=True
        ... )
    """
    diarization = SpeakerDiarization(
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    
    result = diarization.process_call(
        audio_path=audio_path,
        transcription_path=transcription_path,
        call_id=call_id,
        use_simple_heuristic=use_simple_heuristic,
    )
    
    if save_results:
        diarization.save_diarization(
            diarization_result=result,
            output_dir=output_dir,
        )
    
    return result
