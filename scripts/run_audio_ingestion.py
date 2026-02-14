"""
Audio ingestion pipeline runner for EdgeMindCX project.

Processes all .wav files in data/raw/audio/call_center/:
- Converts to 16kHz mono if needed
- Generates call_id for each file
- Saves processed audio to data/processed/audio/{call_id}.wav
"""

import logging
import uuid
from pathlib import Path
from typing import Dict

import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
RAW_AUDIO_DIR = Path("data/raw/audio/call_center")
PROCESSED_AUDIO_DIR = Path("data/processed/audio")
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1  # Mono


def generate_call_id(audio_filename: str) -> str:
    """
    Generate unique call ID from audio filename.
    
    Args:
        audio_filename: Original audio filename.
    
    Returns:
        Unique call ID string.
    """
    # Use UUID + filename stem
    call_id = f"call_{uuid.uuid4().hex[:8]}_{Path(audio_filename).stem}"
    return call_id


def process_audio_file(
    audio_path: Path,
    output_dir: Path,
) -> Dict[str, any]:
    """
    Process a single audio file.
    
    Args:
        audio_path: Path to input audio file.
        output_dir: Directory to save processed audio.
    
    Returns:
        Dictionary with processing results.
    """
    logger.info(f"Processing: {audio_path.name}")
    
    # Generate call_id
    call_id = generate_call_id(audio_path.name)
    logger.info(f"  Generated call_id: {call_id}")
    
    # Get original audio info
    try:
        original_info = sf.info(str(audio_path))
        original_sr = original_info.samplerate
        original_channels = original_info.channels
        original_duration = original_info.duration
        
        logger.info(
            f"  Original: {original_sr}Hz, {original_channels}ch, "
            f"{original_duration:.2f}s"
        )
    except Exception as e:
        logger.error(f"  Error reading audio info: {e}")
        raise
    
    # Load and convert audio
    try:
        logger.info(f"  Loading and converting audio...")
        waveform, sr = librosa.load(
            str(audio_path),
            sr=TARGET_SAMPLE_RATE,
            mono=(TARGET_CHANNELS == 1),
        )
        
        logger.info(f"  Converted: {sr}Hz, {waveform.shape}, {len(waveform)/sr:.2f}s")
    except Exception as e:
        logger.error(f"  Error loading/converting audio: {e}")
        raise
    
    # Save processed audio
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{call_id}.wav"
    
    try:
        logger.info(f"  Saving to: {output_path}")
        sf.write(
            str(output_path),
            waveform,
            TARGET_SAMPLE_RATE,
            subtype="PCM_16",
        )
        
        # Verify saved file
        saved_info = sf.info(str(output_path))
        logger.info(
            f"  Saved: {saved_info.samplerate}Hz, {saved_info.channels}ch, "
            f"{saved_info.duration:.2f}s"
        )
        
    except Exception as e:
        logger.error(f"  Error saving audio: {e}")
        raise
    
    return {
        "call_id": call_id,
        "original_file": str(audio_path),
        "processed_file": str(output_path),
        "original_sample_rate": original_sr,
        "original_channels": original_channels,
        "original_duration": original_duration,
        "processed_sample_rate": saved_info.samplerate,
        "processed_channels": saved_info.channels,
        "processed_duration": saved_info.duration,
        "converted": (
            original_sr != TARGET_SAMPLE_RATE or original_channels != TARGET_CHANNELS
        ),
    }


def run_audio_ingestion() -> None:
    """
    Run audio ingestion pipeline.
    
    Processes all .wav files in raw audio directory and saves
    processed versions to processed audio directory.
    """
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - AUDIO INGESTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Source directory: {RAW_AUDIO_DIR}")
    logger.info(f"Output directory: {PROCESSED_AUDIO_DIR}")
    logger.info(f"Target format: {TARGET_SAMPLE_RATE}Hz, {TARGET_CHANNELS}ch (mono)")
    logger.info("")
    
    # Validate source directory
    if not RAW_AUDIO_DIR.exists():
        logger.error(f"Source directory not found: {RAW_AUDIO_DIR}")
        return
    
    # Find all .wav files
    wav_files = list(RAW_AUDIO_DIR.glob("*.wav"))
    
    if not wav_files:
        logger.warning(f"No .wav files found in: {RAW_AUDIO_DIR}")
        return
    
    logger.info(f"Found {len(wav_files)} audio file(s) to process")
    logger.info("")
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(sorted(wav_files), 1):
        logger.info(f"[{i}/{len(wav_files)}] Processing: {audio_file.name}")
        logger.info("-" * 80)
        
        try:
            result = process_audio_file(audio_file, PROCESSED_AUDIO_DIR)
            results.append(result)
            successful += 1
            logger.info(f"[SUCCESS] Processed: {result['call_id']}")
        except Exception as e:
            logger.error(f"[FAILED] Error processing {audio_file.name}: {e}")
            failed += 1
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(wav_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("")
    
    if results:
        logger.info("Processed files:")
        for result in results:
            logger.info(
                f"  {result['call_id']}: "
                f"{result['original_file']} -> {result['processed_file']}"
            )
            if result['converted']:
                logger.info(
                    f"    Converted: {result['original_sample_rate']}Hz/{result['original_channels']}ch "
                    f"-> {result['processed_sample_rate']}Hz/{result['processed_channels']}ch"
                )
    
    logger.info("=" * 80)
    logger.info("Audio ingestion pipeline completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_audio_ingestion()
