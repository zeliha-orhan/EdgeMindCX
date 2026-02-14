"""
Script to run behavioral analysis pipeline for all calls.

Analyzes:
1. Speech rate (words per minute)
2. Silence duration
3. Pitch and energy trends
4. Speaker-based metrics

Output: data/processed/behavior/{call_id}.csv
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules directly to avoid __init__.py issues
import importlib.util

# Import speaker_dynamics
spec_dynamics = importlib.util.spec_from_file_location(
    "speaker_dynamics_module",
    project_root / "edge_mind_cx" / "analysis" / "speaker_dynamics.py"
)
speaker_dynamics_module = importlib.util.module_from_spec(spec_dynamics)
spec_dynamics.loader.exec_module(speaker_dynamics_module)
SpeakerDynamicsAnalyzer = speaker_dynamics_module.SpeakerDynamicsAnalyzer

# Import audio_features for basic pitch/energy extraction
spec_audio = importlib.util.spec_from_file_location(
    "audio_features_module",
    project_root / "edge_mind_cx" / "analysis" / "audio_features.py"
)
audio_features_module = importlib.util.module_from_spec(spec_audio)
spec_audio.loader.exec_module(audio_features_module)
AudioFeatureExtractor = audio_features_module.AudioFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def combine_behavioral_metrics(
    call_id: str,
    speaker_dynamics_metrics: List[Dict],
    audio_stress_features: pd.DataFrame,
    diarization_data: Dict,
) -> pd.DataFrame:
    """
    Combine all behavioral metrics into a single DataFrame.
    
    Args:
        call_id: Call identifier.
        speaker_dynamics_metrics: List of speaker dynamics metrics.
        audio_stress_features: DataFrame with audio stress features.
        diarization_data: Diarization data with role mapping.
    
    Returns:
        Combined DataFrame with all behavioral metrics.
    """
    # Convert speaker dynamics to DataFrame
    dynamics_df = pd.DataFrame(speaker_dynamics_metrics)
    
    # Get role mapping from diarization
    role_mapping = diarization_data.get("role_mapping", {})
    
    # Add role information to dynamics
    if "speaker" in dynamics_df.columns:
        dynamics_df["role"] = dynamics_df["speaker"].map(
            lambda x: role_mapping.get(x, "unknown")
        )
    
    # Merge with audio stress features if available
    if not audio_stress_features.empty and "speaker_id" in audio_stress_features.columns:
        # Merge on speaker_id
        combined_df = pd.merge(
            dynamics_df,
            audio_stress_features,
            left_on="speaker",
            right_on="speaker_id",
            how="outer",
            suffixes=("_dynamics", "_audio"),
        )
    else:
        combined_df = dynamics_df.copy()
    
    # Add call_id
    combined_df["call_id"] = call_id
    
    # Reorder columns
    priority_columns = ["call_id", "speaker", "role", "metric", "value"]
    other_columns = [c for c in combined_df.columns if c not in priority_columns]
    column_order = priority_columns + other_columns
    
    # Only include columns that exist
    column_order = [c for c in column_order if c in combined_df.columns]
    
    return combined_df[column_order]


def run_behavioral_analysis_pipeline(
    diarization_dir: str | Path = "data/processed/diarization",
    transcripts_dir: str | Path = "data/processed/transcripts",
    processed_audio_dir: str | Path = "data/processed/audio",
    output_behavior_dir: str | Path = "data/processed/behavior",
    fast_path: bool = False,
) -> List[Dict]:
    """
    Run behavioral analysis pipeline for all calls.
    
    Args:
        diarization_dir: Directory containing diarization results.
        transcripts_dir: Directory containing transcription results.
        processed_audio_dir: Directory containing processed audio files.
        output_behavior_dir: Directory to save behavioral analysis results.
        fast_path: If True, skip pitch/energy trends, only calculate speech rate, silence ratio, sentiment.
    
    Returns:
        List of analysis result dictionaries.
    """
    diarization_dir = Path(diarization_dir)
    transcripts_dir = Path(transcripts_dir)
    processed_audio_dir = Path(processed_audio_dir)
    output_behavior_dir = Path(output_behavior_dir)
    
    if not diarization_dir.exists():
        logger.error(f"Diarization directory not found: {diarization_dir}")
        return []
    
    # Find all diarization files
    diarization_files = list(diarization_dir.glob("*.json"))
    
    if not diarization_files:
        logger.warning(f"No diarization files found in {diarization_dir}")
        return []
    
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - BEHAVIORAL ANALYSIS PIPELINE")
    if fast_path:
        logger.info("⚡ FAST PATH MODE: Speech rate, silence ratio, sentiment only")
    logger.info("=" * 80)
    logger.info(f"Diarization directory: {diarization_dir}")
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Processed audio directory: {processed_audio_dir}")
    logger.info(f"Output behavior directory: {output_behavior_dir}")
    logger.info(f"Found {len(diarization_files)} call(s) to analyze")
    logger.info("")
    
    # Create output directory
    output_behavior_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzers
    dynamics_analyzer = SpeakerDynamicsAnalyzer()
    audio_feature_extractor = AudioFeatureExtractor()
    
    all_results = []
    
    for idx, diarization_file in enumerate(diarization_files, 1):
        call_id = diarization_file.stem
        
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(diarization_files)}] Processing: {call_id}")
        logger.info("-" * 80)
        
        try:
            # Load diarization data
            with open(diarization_file, "r", encoding="utf-8") as f:
                diarization_data = json.load(f)
            
            # Find corresponding files
            transcript_file = transcripts_dir / call_id / "segments.json"
            audio_file = processed_audio_dir / f"{call_id}.wav"
            
            if not transcript_file.exists():
                logger.warning(f"Transcript file not found: {transcript_file}")
                continue
            
            if not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                continue
            
            # 1. Speaker Dynamics Analysis (speech rate, silence)
            logger.info("Analyzing speaker dynamics...")
            
            # Load transcription data
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcription_data = json.load(f)
            
            dynamics_metrics = dynamics_analyzer.analyze_call(
                call_id=call_id,
                diarization_data=diarization_data,
                transcription_data=transcription_data,
            )
            
            # 2. Audio Stress Features (pitch, energy) - Skip in fast path
            audio_features_df = pd.DataFrame()
            
            if not fast_path:
                logger.info("Extracting audio stress features (pitch, energy)...")
                try:
                    # Use simplified approach: extract pitch and energy per speaker segment
                    import librosa
                    import numpy as np
                    
                    # Load full audio
                    audio_array, sample_rate = librosa.load(str(audio_file), sr=16000, mono=True)
                    
                    # Extract features for each speaker segment
                    speaker_features = []
                    for segment in diarization_data.get("diarization_segments", []):
                        speaker_id = segment.get("speaker_id", "UNKNOWN")
                        start_time = segment.get("start_time", 0.0)
                        end_time = segment.get("end_time", 0.0)
                        
                        # Extract segment
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        segment_audio = audio_array[start_sample:end_sample]
                        
                        if len(segment_audio) < 100:  # Skip very short segments
                            continue
                        
                        # Extract pitch (F0)
                        try:
                            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sample_rate)
                            pitch_values = []
                            for t in range(pitches.shape[1]):
                                index = magnitudes[:, t].argmax()
                                pitch = pitches[index, t]
                                if pitch > 0:
                                    pitch_values.append(pitch)
                            
                            pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
                            pitch_std = float(np.std(pitch_values)) if pitch_values else 0.0
                        except:
                            pitch_mean = 0.0
                            pitch_std = 0.0
                        
                        # Extract energy (RMS)
                        try:
                            rms = librosa.feature.rms(y=segment_audio)[0]
                            energy_mean = float(np.mean(rms))
                            energy_std = float(np.std(rms))
                        except:
                            energy_mean = 0.0
                            energy_std = 0.0
                        
                        speaker_features.append({
                            "speaker_id": speaker_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "pitch_mean": pitch_mean,
                            "pitch_std": pitch_std,
                            "energy_mean": energy_mean,
                            "energy_std": energy_std,
                        })
                    
                    if speaker_features:
                        audio_features_df = pd.DataFrame(speaker_features)
                        # Aggregate by speaker
                        audio_features_agg = audio_features_df.groupby("speaker_id").agg({
                            "pitch_mean": ["mean", "std"],
                            "pitch_std": "mean",
                            "energy_mean": ["mean", "std"],
                            "energy_std": "mean",
                        }).reset_index()
                        audio_features_agg.columns = ["speaker_id", "pitch_mean_avg", "pitch_mean_std", "pitch_std_avg", "energy_mean_avg", "energy_mean_std", "energy_std_avg"]
                        audio_features_df = audio_features_agg
                        
                except Exception as e:
                    logger.warning(f"Audio stress feature extraction failed: {e}")
                    audio_features_df = pd.DataFrame()
            else:
                logger.info("⚡ Fast path: Skipping pitch/energy trend extraction")
            
            # 3. Combine all metrics
            logger.info("Combining behavioral metrics...")
            combined_df = combine_behavioral_metrics(
                call_id=call_id,
                speaker_dynamics_metrics=dynamics_metrics,
                audio_stress_features=audio_features_df,
                diarization_data=diarization_data,
            )
            
            # 4. Save to CSV
            output_file = output_behavior_dir / f"{call_id}.csv"
            combined_df.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"Saved behavioral analysis: {output_file}")
            
            # Print summary
            logger.info("")
            logger.info(f"[SUMMARY] Call ID: {call_id}")
            logger.info(f"  Total metrics: {len(combined_df)}")
            logger.info(f"  Unique speakers: {combined_df['speaker'].nunique() if 'speaker' in combined_df.columns else 'N/A'}")
            logger.info(f"  Metrics: {combined_df['metric'].nunique() if 'metric' in combined_df.columns else 'N/A'}")
            logger.info(f"  Output file: {output_file}")
            logger.info("")
            
            all_results.append({
                "call_id": call_id,
                "output_file": str(output_file),
                "num_metrics": len(combined_df),
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {call_id}: {e}")
            logger.exception(e)
            continue
    
    # Final summary
    logger.info("=" * 80)
    logger.info("BEHAVIORAL ANALYSIS PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total calls processed: {len(all_results)}/{len(diarization_files)}")
    
    if all_results:
        total_metrics = sum(r.get("num_metrics", 0) for r in all_results)
        logger.info(f"Total metrics generated: {total_metrics}")
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run behavioral analysis pipeline
    results = run_behavioral_analysis_pipeline(
        diarization_dir="data/processed/diarization",
        transcripts_dir="data/processed/transcripts",
        processed_audio_dir="data/processed/audio",
        output_behavior_dir="data/processed/behavior",
    )
    
    if results:
        logger.info("\n✓ Behavioral analysis pipeline completed successfully!")
    else:
        logger.warning("\n⚠ No calls were processed.")
