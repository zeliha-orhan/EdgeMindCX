"""
Whisper transcription pipeline for EdgeMindCX project.

Processes validated audio files through Whisper transcription with word-level
timestamps and saves results in structured format.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import whisper

logger = logging.getLogger(__name__)

# Global cache for Whisper models (singleton pattern)
_whisper_model_cache: Dict[str, any] = {}


class WhisperTranscriptionPipeline:
    """
    Pipeline for transcribing validated audio files using Whisper.
    
    Features:
    - Local Whisper model
    - Word-level timestamps
    - Segment-based JSON output
    - Raw transcript TXT output
    - Organized file structure: data/processed/transcripts/{call_id}/
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        language: Optional[str] = None,
        output_base_dir: str | Path = "data/processed/transcripts",
        word_timestamps: bool = True,
    ) -> None:
        """
        Initialize Whisper transcription pipeline.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                       Default is 'small'.
            device: Device to run model on ('cpu' or 'cuda'). If None, auto-detects.
            language: Language code (e.g., 'en', 'tr'). If None, auto-detects.
            output_base_dir: Base directory for transcript outputs.
                           Default is 'data/processed/transcripts'.
            word_timestamps: Whether to include word-level timestamps. Default is True.
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.output_base_dir = Path(output_base_dir)
        self.word_timestamps = word_timestamps
        
        # Initialize Whisper model (use global cache to avoid reloading)
        cache_key = f"{model_size}_{device or 'auto'}"
        
        if cache_key in _whisper_model_cache:
            logger.info(f"Using cached Whisper model: {model_size}")
            self.model = _whisper_model_cache[cache_key]
        else:
            logger.info(f"Loading Whisper model: {model_size} (first time, will be cached)")
            self.model = whisper.load_model(model_size, device=device)
            _whisper_model_cache[cache_key] = self.model
            logger.info(f"Whisper model loaded and cached successfully")
        
        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_base_dir}")
    
    def _generate_call_id(self, audio_path: Path) -> str:
        """
        Generate unique call ID from audio file path.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Unique call ID string.
        """
        # Use filename stem as base, add UUID for uniqueness
        call_id = f"call_{uuid.uuid4().hex[:8]}_{audio_path.stem}"
        return call_id
    
    def transcribe_audio(
        self,
        audio_path: str | Path,
        call_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file.
            call_id: Optional call ID. If None, will be generated automatically.
        
        Returns:
            Dictionary containing transcription results:
            - call_id: Unique call identifier
            - audio_path: Path to audio file
            - text: Full transcribed text
            - segments: List of segments with word-level timestamps
            - language: Detected language
            - duration: Audio duration in seconds
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate call_id if not provided
        if call_id is None:
            call_id = self._generate_call_id(audio_path)
        
        logger.info(f"Transcribing audio: {audio_path.name} (Call ID: {call_id})")
        
        # Load audio using librosa (to avoid ffmpeg dependency)
        logger.debug(f"Loading audio file: {audio_path}")
        audio_array, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        logger.debug(f"Audio loaded: {len(audio_array)} samples, {sample_rate}Hz")
        
        # Prepare transcription options
        transcribe_options = {
            "word_timestamps": self.word_timestamps,
        }
        
        if self.language:
            transcribe_options["language"] = self.language
        
        # Transcribe using Whisper (pass numpy array directly)
        logger.debug(f"Running Whisper transcription with word_timestamps={self.word_timestamps}")
        result = self.model.transcribe(
            audio_array,
            **transcribe_options,
        )
        
        # Extract segments with word-level timestamps
        segments = []
        for segment in result.get("segments", []):
            segment_data = {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
            }
            
            # Add word-level timestamps if available
            if self.word_timestamps and "words" in segment:
                segment_data["words"] = [
                    {
                        "word": word.get("word", "").strip(),
                        "start": word.get("start"),
                        "end": word.get("end"),
                    }
                    for word in segment.get("words", [])
                ]
            
            segments.append(segment_data)
        
        transcription_result = {
            "call_id": call_id,
            "audio_path": str(audio_path),
            "filename": audio_path.name,
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
        }
        
        logger.info(
            f"Transcription completed: {len(segments)} segments, "
            f"language={transcription_result['language']}, "
            f"duration={transcription_result['duration']:.2f}s"
        )
        
        return transcription_result
    
    def save_transcription(
        self,
        transcription_result: Dict[str, any],
    ) -> Dict[str, Path]:
        """
        Save transcription results to files.
        
        Saves:
        - Segment-based JSON: {call_id}/segments.json
        - Raw transcript TXT: {call_id}/transcript.txt
        
        Args:
            transcription_result: Transcription result dictionary.
        
        Returns:
            Dictionary with paths to saved files:
            - json_path: Path to JSON file
            - txt_path: Path to TXT file
        """
        call_id = transcription_result["call_id"]
        call_dir = self.output_base_dir / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving transcription for call_id: {call_id}")
        
        # Save segment-based JSON
        json_path = call_dir / "segments.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcription_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON transcript: {json_path}")
        
        # Save raw transcript TXT
        txt_path = call_dir / "transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcription_result["text"])
        logger.info(f"Saved TXT transcript: {txt_path}")
        
        return {
            "json_path": json_path,
            "txt_path": txt_path,
            "call_dir": call_dir,
        }
    
    def process_audio_file(
        self,
        audio_path: str | Path,
        call_id: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, any]:
        """
        Process a single audio file through complete transcription pipeline.
        
        Args:
            audio_path: Path to audio file.
            call_id: Optional call ID. If None, will be generated automatically.
            save_results: Whether to save results to files. Default is True.
        
        Returns:
            Dictionary containing transcription results and file paths.
        """
        # Transcribe
        transcription_result = self.transcribe_audio(
            audio_path=audio_path,
            call_id=call_id,
        )
        
        # Save results
        saved_paths = {}
        if save_results:
            saved_paths = self.save_transcription(transcription_result)
        
        return {
            **transcription_result,
            "saved_paths": saved_paths,
        }
    
    def process_validated_audio_list(
        self,
        validated_audio_list: List[Dict[str, any]],
        save_results: bool = True,
    ) -> List[Dict[str, any]]:
        """
        Process a list of validated audio files through transcription pipeline.
        
        Args:
            validated_audio_list: List of validated audio dictionaries from ingestion.
            save_results: Whether to save results to files. Default is True.
        
        Returns:
            List of transcription result dictionaries.
        """
        logger.info(f"Processing {len(validated_audio_list)} audio files...")
        
        all_results = []
        
        for idx, audio_info in enumerate(validated_audio_list, 1):
            # Skip files with errors
            if audio_info.get("status") == "error":
                logger.warning(
                    f"Skipping file {idx}/{len(validated_audio_list)}: "
                    f"{audio_info.get('filename')} (error status)"
                )
                continue
            
            audio_path = Path(audio_info["file_path"])
            
            try:
                logger.info(f"Processing file {idx}/{len(validated_audio_list)}: {audio_path.name}")
                
                result = self.process_audio_file(
                    audio_path=audio_path,
                    save_results=save_results,
                )
                
                all_results.append(result)
                
                logger.info(
                    f"✓ Completed {idx}/{len(validated_audio_list)}: "
                    f"{result['call_id']} ({len(result['segments'])} segments)"
                )
                
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                continue
        
        # Log summary
        self._log_summary(all_results)
        
        return all_results
    
    def _log_summary(self, results: List[Dict[str, any]]) -> None:
        """
        Log summary statistics of transcription results.
        
        Args:
            results: List of transcription result dictionaries.
        """
        total = len(results)
        total_segments = sum(len(r.get("segments", [])) for r in results)
        total_duration = sum(r.get("duration", 0.0) for r in results)
        total_words = sum(
            sum(len(seg.get("words", [])) for seg in r.get("segments", []))
            for r in results
        )
        
        languages = {}
        for r in results:
            lang = r.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        
        logger.info("=" * 60)
        logger.info("TRANSCRIPTION PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {total}")
        logger.info(f"Total segments: {total_segments}")
        logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        logger.info(f"Total words: {total_words}")
        logger.info(f"Languages detected: {languages}")
        logger.info("=" * 60)


def transcribe_validated_audio(
    validated_audio_list: List[Dict[str, any]],
    model_size: str = "small",
    output_base_dir: str | Path = "data/processed/transcripts",
    word_timestamps: bool = True,
    save_results: bool = True,
) -> List[Dict[str, any]]:
    """
    Convenience function to transcribe validated audio files.
    
    Args:
        validated_audio_list: List of validated audio dictionaries from ingestion.
        model_size: Whisper model size. Default is 'small'.
        output_base_dir: Base directory for transcript outputs.
        word_timestamps: Whether to include word-level timestamps. Default is True.
        save_results: Whether to save results to files. Default is True.
    
    Returns:
        List of transcription result dictionaries.
    
    Example:
        >>> from edge_mind_cx.audio import ingest_audio_files
        >>> from edge_mind_cx.transcription.pipeline import transcribe_validated_audio
        >>> 
        >>> validated_list = ingest_audio_files()
        >>> transcripts = transcribe_validated_audio(validated_list)
    """
    pipeline = WhisperTranscriptionPipeline(
        model_size=model_size,
        output_base_dir=output_base_dir,
        word_timestamps=word_timestamps,
    )
    
    return pipeline.process_validated_audio_list(
        validated_audio_list=validated_audio_list,
        save_results=save_results,
    )
