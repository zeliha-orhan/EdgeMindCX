"""
Script to run speaker diarization pipeline for transcribed calls.

Processes all transcribed calls and generates:
- Speaker segments with timestamps
- Alignment with transcription
- Agent and customer speaking durations
- Output: data/processed/diarization/{call_id}.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from module to avoid __init__.py issues
# Note: This will fail if pyannote.audio dependencies have issues
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diarization_module",
        project_root / "edge_mind_cx" / "transcription" / "diarization.py"
    )
    diarization_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diarization_module)
    SpeakerDiarization = diarization_module.SpeakerDiarization
except RuntimeError as e:
    if "torchvision" in str(e) or "torch" in str(e):
        logger.error("=" * 80)
        logger.error("ERROR: Torch/Torchvision compatibility issue detected.")
        logger.error("This is a known issue with pyannote.audio dependencies.")
        logger.error("=" * 80)
        logger.error("Possible solutions:")
        logger.error("1. Update torch and torchvision to compatible versions")
        logger.error("2. Reinstall pyannote.audio: pip install --upgrade pyannote.audio")
        logger.error("3. Use a different Python environment")
        logger.error("=" * 80)
        raise
    else:
        raise


def identify_agent_customer(
    speakers: List[str],
    speaker_durations: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Identify which speaker is agent and which is customer.
    
    Assumes agent speaks more (longer total duration).
    
    Args:
        speakers: List of speaker IDs.
        speaker_durations: Dictionary with speaker durations.
    
    Returns:
        Dictionary mapping speaker_id to role ("agent" or "customer").
    """
    if len(speakers) < 2:
        # If only one speaker, assume it's the agent
        return {speakers[0]: "agent"} if speakers else {}
    
    # Sort speakers by total duration (descending)
    speaker_total_durations = [
        (sp, speaker_durations.get(sp, {}).get("total_duration", 0.0))
        for sp in speakers
    ]
    speaker_total_durations.sort(key=lambda x: x[1], reverse=True)
    
    # Agent is the one who speaks more
    role_mapping = {}
    if len(speaker_total_durations) >= 1:
        role_mapping[speaker_total_durations[0][0]] = "agent"
    if len(speaker_total_durations) >= 2:
        role_mapping[speaker_total_durations[1][0]] = "customer"
    
    # Assign remaining speakers as "unknown"
    for sp in speakers:
        if sp not in role_mapping:
            role_mapping[sp] = "unknown"
    
    return role_mapping


def run_diarization_pipeline(
    transcripts_dir: str | Path = "data/processed/transcripts",
    processed_audio_dir: str | Path = "data/processed/audio",
    output_diarization_dir: str | Path = "data/processed/diarization",
    min_speakers: int = 2,
    max_speakers: int | None = None,
    pyannote_auth_token: str | None = None,
    use_simple_heuristic: bool = False,
) -> List[Dict]:
    """
    Run speaker diarization pipeline for all transcribed calls.
    
    Args:
        transcripts_dir: Directory containing transcription results.
        processed_audio_dir: Directory containing processed audio files.
        output_diarization_dir: Directory to save diarization results.
        min_speakers: Minimum number of speakers. Default is 2.
        max_speakers: Maximum number of speakers. If None, auto-detects.
        pyannote_auth_token: HuggingFace auth token for pyannote models.
        use_simple_heuristic: If True, use simple 2-speaker heuristic instead of pyannote (fast path).
    
    Returns:
        List of diarization result dictionaries.
    """
    transcripts_dir = Path(transcripts_dir)
    processed_audio_dir = Path(processed_audio_dir)
    output_diarization_dir = Path(output_diarization_dir)
    
    if not transcripts_dir.exists():
        logger.error(f"Transcripts directory not found: {transcripts_dir}")
        return []
    
    # Find all call directories with segments.json
    call_dirs = [
        d for d in transcripts_dir.iterdir()
        if d.is_dir() and (d / "segments.json").exists()
    ]
    
    if not call_dirs:
        logger.warning(f"No transcribed calls found in {transcripts_dir}")
        return []
    
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - SPEAKER DIARIZATION PIPELINE")
    if use_simple_heuristic:
        logger.info("⚡ FAST PATH MODE: Using simple 2-speaker heuristic")
    logger.info("=" * 80)
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Processed audio directory: {processed_audio_dir}")
    logger.info(f"Output diarization directory: {output_diarization_dir}")
    logger.info(f"Min speakers: {min_speakers}, Max speakers: {max_speakers}")
    logger.info(f"Found {len(call_dirs)} call(s) to process")
    logger.info("")
    
    # Initialize diarization pipeline (only if not using simple heuristic)
    diarization = None
    if not use_simple_heuristic:
        try:
            diarization = SpeakerDiarization(
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                use_auth_token=pyannote_auth_token,
            )
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            logger.error("Make sure pyannote.audio is installed and configured correctly.")
            return []
    
    all_results = []
    
    for idx, call_dir in enumerate(call_dirs, 1):
        call_id = call_dir.name
        
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(call_dirs)}] Processing: {call_id}")
        logger.info("-" * 80)
        
        # Find corresponding audio file
        audio_file = processed_audio_dir / f"{call_id}.wav"
        if not audio_file.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            logger.warning(f"Skipping {call_id}")
            continue
        
        # Load transcription segments
        segments_file = call_dir / "segments.json"
        try:
            with open(segments_file, "r", encoding="utf-8") as f:
                transcription_data = json.load(f)
                transcription_segments = transcription_data.get("segments", [])
        except Exception as e:
            logger.error(f"Failed to load transcription: {e}")
            continue
        
        try:
            # Perform diarization
            if use_simple_heuristic:
                # Use simple heuristic: create temporary diarization object just for process_call
                temp_diarization = SpeakerDiarization(
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    use_auth_token=pyannote_auth_token,
                    use_simple_heuristic_only=True,  # Skip pyannote initialization
                )
                result = temp_diarization.process_call(
                    audio_path=audio_file,
                    transcription_path=segments_file,
                    call_id=call_id,
                    use_simple_heuristic=True,
                )
                save_obj = temp_diarization
            else:
                result = diarization.process_call(
                    audio_path=audio_file,
                    transcription_path=segments_file,
                    call_id=call_id,
                    use_simple_heuristic=False,
                )
                save_obj = diarization
            
            # Identify agent and customer
            role_mapping = identify_agent_customer(
                result["speakers"],
                result["speaker_durations"],
            )
            result["role_mapping"] = role_mapping
            
            # Add agent and customer speaking durations
            agent_duration = 0.0
            customer_duration = 0.0
            
            for speaker_id, role in role_mapping.items():
                duration_info = result["speaker_durations"].get(speaker_id, {})
                total_duration = duration_info.get("total_duration", 0.0)
                
                if role == "agent":
                    agent_duration = total_duration
                elif role == "customer":
                    customer_duration = total_duration
            
            result["agent_duration"] = agent_duration
            result["customer_duration"] = customer_duration
            result["total_call_duration"] = agent_duration + customer_duration
            
            # Save results
            output_file = save_obj.save_diarization(
                diarization_result=result,
                output_dir=output_diarization_dir,
                filename=f"{call_id}.json",
            )
            
            all_results.append(result)
            
            # Print summary
            logger.info("")
            logger.info(f"[SUMMARY] Call ID: {call_id}")
            logger.info(f"  Audio: {audio_file.name}")
            logger.info(f"  Speakers detected: {result['num_speakers']}")
            logger.info(f"  Speaker IDs: {result['speakers']}")
            logger.info(f"  Role mapping: {role_mapping}")
            logger.info(f"  Diarization segments: {len(result['diarization_segments'])}")
            logger.info(f"  Aligned segments: {len(result['aligned_segments'])}")
            logger.info(f"  Agent speaking duration: {agent_duration:.2f}s")
            logger.info(f"  Customer speaking duration: {customer_duration:.2f}s")
            logger.info(f"  Total call duration: {result['total_call_duration']:.2f}s")
            logger.info(f"  Output file: {output_file}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {call_id}: {e}")
            logger.exception(e)
            continue
    
    # Final summary
    logger.info("=" * 80)
    logger.info("DIARIZATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total calls processed: {len(all_results)}/{len(call_dirs)}")
    
    if all_results:
        total_agent_duration = sum(r.get("agent_duration", 0.0) for r in all_results)
        total_customer_duration = sum(r.get("customer_duration", 0.0) for r in all_results)
        total_segments = sum(len(r.get("diarization_segments", [])) for r in all_results)
        
        logger.info(f"Total diarization segments: {total_segments}")
        logger.info(f"Total agent speaking time: {total_agent_duration:.2f}s ({total_agent_duration/60:.2f} minutes)")
        logger.info(f"Total customer speaking time: {total_customer_duration:.2f}s ({total_customer_duration/60:.2f} minutes)")
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run diarization pipeline
    # HuggingFace token for pyannote.audio: set HF_TOKEN or PYANNOTE_AUTH_TOKEN in env
    import os
    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_AUTH_TOKEN")
    
    results = run_diarization_pipeline(
        transcripts_dir="data/processed/transcripts",
        processed_audio_dir="data/processed/audio",
        output_diarization_dir="data/processed/diarization",
        min_speakers=2,
        max_speakers=None,
        pyannote_auth_token=HF_TOKEN,
    )
    
    if results:
        logger.info("\n✓ Diarization pipeline completed successfully!")
    else:
        logger.warning("\n⚠ No calls were processed.")
