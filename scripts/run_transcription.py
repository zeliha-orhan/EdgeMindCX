"""
Script to run Whisper transcription pipeline for processed audio files.

Processes all .wav files in data/processed/audio/ and generates:
- segments.json (with word-level timestamps)
- transcript.txt (raw text)
"""

import logging
import sys
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from module file to avoid __init__.py dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "transcription_pipeline",
    project_root / "edge_mind_cx" / "transcription" / "pipeline.py"
)
transcription_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcription_module)
WhisperTranscriptionPipeline = transcription_module.WhisperTranscriptionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_transcription_pipeline(
    processed_audio_dir: str | Path = "data/processed/audio",
    output_transcripts_dir: str | Path = "data/processed/transcripts",
    whisper_model_size: str = "small",
    word_timestamps: bool = True,
) -> List[dict]:
    """
    Run Whisper transcription pipeline for all processed audio files.
    
    Args:
        processed_audio_dir: Directory containing processed audio files.
        output_transcripts_dir: Directory to save transcription results.
        whisper_model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        word_timestamps: Whether to include word-level timestamps.
    
    Returns:
        List of transcription result dictionaries.
    """
    processed_audio_dir = Path(processed_audio_dir)
    
    if not processed_audio_dir.exists():
        logger.error(f"Processed audio directory not found: {processed_audio_dir}")
        return []
    
    # Find all .wav files
    audio_files = list(processed_audio_dir.glob("*.wav"))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {processed_audio_dir}")
        return []
    
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - WHISPER TRANSCRIPTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Processed audio directory: {processed_audio_dir}")
    logger.info(f"Output transcripts directory: {output_transcripts_dir}")
    logger.info(f"Whisper model: {whisper_model_size}")
    logger.info(f"Word-level timestamps: {word_timestamps}")
    logger.info(f"Found {len(audio_files)} audio file(s) to transcribe")
    logger.info("")
    
    # Initialize transcription pipeline
    pipeline = WhisperTranscriptionPipeline(
        model_size=whisper_model_size,
        output_base_dir=output_transcripts_dir,
        word_timestamps=word_timestamps,
    )
    
    # Process each audio file
    all_results = []
    
    for idx, audio_path in enumerate(audio_files, 1):
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(audio_files)}] Processing: {audio_path.name}")
        logger.info("-" * 80)
        
        # Extract call_id from filename (remove .wav extension)
        call_id = audio_path.stem
        
        try:
            # Transcribe audio
            result = pipeline.process_audio_file(
                audio_path=audio_path,
                call_id=call_id,
                save_results=True,
            )
            
            all_results.append(result)
            
            # Print summary for this call
            logger.info("")
            logger.info(f"[SUMMARY] Call ID: {call_id}")
            logger.info(f"  Audio: {audio_path.name}")
            logger.info(f"  Duration: {result['duration']:.2f}s")
            logger.info(f"  Language: {result['language']}")
            logger.info(f"  Segments: {len(result['segments'])}")
            logger.info(f"  Total words: {sum(len(seg.get('words', [])) for seg in result['segments'])}")
            logger.info(f"  Output directory: {result['saved_paths']['call_dir']}")
            logger.info(f"  Files saved:")
            logger.info(f"    - {result['saved_paths']['json_path'].name}")
            logger.info(f"    - {result['saved_paths']['txt_path'].name}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to transcribe {audio_path.name}: {e}")
            logger.exception(e)
            continue
    
    # Final summary
    logger.info("=" * 80)
    logger.info("TRANSCRIPTION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(all_results)}/{len(audio_files)}")
    
    if all_results:
        total_segments = sum(len(r.get("segments", [])) for r in all_results)
        total_duration = sum(r.get("duration", 0.0) for r in all_results)
        total_words = sum(
            sum(len(seg.get("words", [])) for seg in r.get("segments", []))
            for r in all_results
        )
        
        logger.info(f"Total segments: {total_segments}")
        logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        logger.info(f"Total words: {total_words}")
        
        languages = {}
        for r in all_results:
            lang = r.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        logger.info(f"Languages detected: {languages}")
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run transcription pipeline
    results = run_transcription_pipeline(
        processed_audio_dir="data/processed/audio",
        output_transcripts_dir="data/processed/transcripts",
        whisper_model_size="small",
        word_timestamps=True,
    )
    
    if results:
        logger.info("\n✓ Transcription pipeline completed successfully!")
    else:
        logger.warning("\n⚠ No files were transcribed.")
