"""
Script to calculate behavioral scores from behavior and transcript data.

Calculates:
1. Stress tendency score
2. Agent-customer empathy alignment
3. Early churn risk
4. Explainable heuristic scores

Output: JSON files with all scores and explanations
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules directly to avoid __init__.py issues
import importlib.util

# Import stress score fusion
spec_stress = importlib.util.spec_from_file_location(
    "stress_fusion_module",
    project_root / "edge_mind_cx" / "scoring" / "stress_score_fusion.py"
)
stress_module = importlib.util.module_from_spec(spec_stress)
spec_stress.loader.exec_module(stress_module)
StressScoreFusion = stress_module.StressScoreFusion

# Import empathy convergence
spec_empathy = importlib.util.spec_from_file_location(
    "empathy_convergence_module",
    project_root / "edge_mind_cx" / "behavioral" / "empathy_convergence.py"
)
empathy_module = importlib.util.module_from_spec(spec_empathy)
spec_empathy.loader.exec_module(empathy_module)
EmpathyConvergenceAnalyzer = empathy_module.EmpathyConvergenceAnalyzer

# Import early churn risk
spec_churn = importlib.util.spec_from_file_location(
    "early_churn_risk_module",
    project_root / "edge_mind_cx" / "behavioral" / "early_churn_risk.py"
)
churn_module = importlib.util.module_from_spec(spec_churn)
spec_churn.loader.exec_module(churn_module)
EarlyChurnRiskAnalyzer = churn_module.EarlyChurnRiskAnalyzer

# Import text stress analyzer
spec_text = importlib.util.spec_from_file_location(
    "text_stress_module",
    project_root / "edge_mind_cx" / "behavioral" / "text_stress_analyzer.py"
)
text_module = importlib.util.module_from_spec(spec_text)
spec_text.loader.exec_module(text_module)
TextStressAnalyzer = text_module.TextStressAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_audio_features_from_behavior(
    behavior_df: pd.DataFrame,
    speaker_id: str,
) -> Dict[str, float]:
    """
    Extract audio features from behavior CSV for a specific speaker.
    
    Args:
        behavior_df: Behavior metrics DataFrame.
        speaker_id: Speaker ID to extract features for.
    
    Returns:
        Dictionary of audio features.
    """
    # Filter for speaker
    speaker_data = behavior_df[behavior_df.get("speaker_id") == speaker_id]
    
    if speaker_data.empty:
        return {}
    
    # Extract pitch and energy features
    features = {}
    
    # Pitch features
    if "pitch_mean_avg" in speaker_data.columns:
        pitch_mean = speaker_data["pitch_mean_avg"].iloc[0] if not speaker_data.empty else 0.0
        pitch_std = speaker_data["pitch_mean_std"].iloc[0] if not speaker_data.empty else 0.0
        features["pitch_mean"] = float(pitch_mean)
        features["pitch_variance"] = float(pitch_std ** 2) if pitch_std else 0.0
    
    # Energy features
    if "energy_mean_avg" in speaker_data.columns:
        energy_mean = speaker_data["energy_mean_avg"].iloc[0] if not speaker_data.empty else 0.0
        energy_std = speaker_data["energy_mean_std"].iloc[0] if not speaker_data.empty else 0.0
        features["energy_rms_mean"] = float(energy_mean)
        features["energy_rms_std"] = float(energy_std)
    
    # Speech rate from behavior metrics
    speech_rate_row = behavior_df[
        (behavior_df["speaker"] == speaker_id.replace("SPEAKER_", "").lower()) &
        (behavior_df["metric"] == "words_per_minute")
    ]
    if not speech_rate_row.empty:
        features["speech_rate"] = float(speech_rate_row["value"].iloc[0])
    
    # Silence ratio
    silence_row = behavior_df[
        (behavior_df["speaker"] == speaker_id.replace("SPEAKER_", "").lower()) &
        (behavior_df["metric"] == "total_silence_seconds")
    ]
    speaking_time_row = behavior_df[
        (behavior_df["speaker"] == speaker_id.replace("SPEAKER_", "").lower()) &
        (behavior_df["metric"] == "total_speaking_time_seconds")
    ]
    if not silence_row.empty and not speaking_time_row.empty:
        total_time = float(silence_row["value"].iloc[0]) + float(speaking_time_row["value"].iloc[0])
        if total_time > 0:
            features["silence_ratio"] = float(silence_row["value"].iloc[0]) / total_time
        else:
            features["silence_ratio"] = 0.0
    
    return features


def calculate_stress_tendency(
    call_id: str,
    behavior_df: pd.DataFrame,
    transcript_text: str,
    diarization_data: Dict,
) -> Dict[str, any]:
    """
    Calculate stress tendency score from behavior and transcript.
    
    Args:
        call_id: Call identifier.
        behavior_df: Behavior metrics DataFrame.
        transcript_text: Full transcript text.
        diarization_data: Diarization data with role mapping.
    
    Returns:
        Dictionary with stress scores and explanations.
    """
    logger.info(f"Calculating stress tendency for {call_id}...")
    
    # Initialize analyzers
    stress_fusion = StressScoreFusion()
    text_analyzer = TextStressAnalyzer()
    
    # Get role mapping
    role_mapping = diarization_data.get("role_mapping", {})
    
    # Find agent and customer speaker IDs
    agent_speaker_id = None
    customer_speaker_id = None
    for speaker_id, role in role_mapping.items():
        if role == "agent":
            agent_speaker_id = speaker_id
        elif role == "customer":
            customer_speaker_id = speaker_id
    
    # Extract audio features for customer (stress is primarily customer-focused)
    customer_audio_features = {}
    if customer_speaker_id:
        customer_audio_features = extract_audio_features_from_behavior(
            behavior_df, customer_speaker_id
        )
    
    # Get text stress score
    text_stress_result = text_analyzer.analyze(transcript_text)
    nlp_stress_score = text_stress_result.get("text_stress_score", 0.0)
    
    # Calculate unified stress score
    stress_result = stress_fusion.compute_final_score(
        audio_features=customer_audio_features if customer_audio_features else None,
        opensmile_features=None,  # Can be added if available
        nlp_stress_score=nlp_stress_score,
    )
    
    # Calculate stress trend (increasing/decreasing)
    stress_tendency = "stable"
    stress_score = stress_result.get("stress_score", 0.0)
    
    if stress_score > 0.7:
        stress_tendency = "high"
    elif stress_score > 0.4:
        stress_tendency = "moderate"
    else:
        stress_tendency = "low"
    
    return {
        "call_id": call_id,
        "stress_score": float(stress_score),
        "stress_tendency": stress_tendency,
        "component_scores": {
            "audio_stress": float(stress_result.get("audio_stress_score", 0.0)),
            "nlp_stress": float(nlp_stress_score),
            "text_negativity": float(text_stress_result.get("negativity_score", 0.0)),
            "text_intensity": float(text_stress_result.get("intensity_score", 0.0)),
        },
        "explanation": (
            f"Stress score: {stress_score:.2f} ({stress_tendency}). "
            f"Audio features contribute {stress_result.get('audio_stress_score', 0.0):.2f}, "
            f"text analysis contributes {nlp_stress_score:.2f}. "
            f"Negativity: {text_stress_result.get('negativity_score', 0.0):.2f}, "
            f"Intensity: {text_stress_result.get('intensity_score', 0.0):.2f}."
        ),
    }


def calculate_empathy_alignment(
    call_id: str,
    behavior_df: pd.DataFrame,
    diarization_data: Dict,
) -> Dict[str, any]:
    """
    Calculate agent-customer empathy alignment.
    
    Args:
        call_id: Call identifier.
        behavior_df: Behavior metrics DataFrame.
        diarization_data: Diarization data with role mapping.
    
    Returns:
        Dictionary with empathy scores and explanations.
    """
    logger.info(f"Calculating empathy alignment for {call_id}...")
    
    # Get role mapping
    role_mapping = diarization_data.get("role_mapping", {})
    
    # Find agent and customer speaker IDs
    agent_speaker_id = None
    customer_speaker_id = None
    for speaker_id, role in role_mapping.items():
        if role == "agent":
            agent_speaker_id = speaker_id
        elif role == "customer":
            customer_speaker_id = speaker_id
    
    if not agent_speaker_id or not customer_speaker_id:
        logger.warning(f"Could not identify agent/customer speakers for {call_id}")
        return {
            "call_id": call_id,
            "empathy_alignment_score": 50.0,
            "explanation": "Could not identify agent/customer speakers",
        }
    
    # Extract features for both speakers
    agent_features = extract_audio_features_from_behavior(behavior_df, agent_speaker_id)
    customer_features = extract_audio_features_from_behavior(behavior_df, customer_speaker_id)
    
    # Get speech rates
    agent_speech_rate = agent_features.get("speech_rate", 0.0)
    customer_speech_rate = customer_features.get("speech_rate", 0.0)
    
    # Get pitch means
    agent_pitch = agent_features.get("pitch_mean", 0.0)
    customer_pitch = customer_features.get("pitch_mean", 0.0)
    
    # Get energy means
    agent_energy = agent_features.get("energy_rms_mean", 0.0)
    customer_energy = customer_features.get("energy_rms_mean", 0.0)
    
    # Calculate alignment scores
    speech_rate_alignment = 1.0 - abs(agent_speech_rate - customer_speech_rate) / max(agent_speech_rate + customer_speech_rate, 1.0)
    speech_rate_alignment = max(0.0, min(1.0, speech_rate_alignment))
    
    pitch_alignment = 1.0 - abs(agent_pitch - customer_pitch) / max(agent_pitch + customer_pitch, 1.0)
    pitch_alignment = max(0.0, min(1.0, pitch_alignment))
    
    energy_alignment = 1.0 - abs(agent_energy - customer_energy) / max(agent_energy + customer_energy, 0.001)
    energy_alignment = max(0.0, min(1.0, energy_alignment))
    
    # Weighted empathy score (0-100)
    empathy_score = (
        0.35 * speech_rate_alignment +
        0.35 * pitch_alignment +
        0.30 * energy_alignment
    ) * 100.0
    
    # Determine alignment level
    if empathy_score >= 75:
        alignment_level = "high"
    elif empathy_score >= 50:
        alignment_level = "moderate"
    else:
        alignment_level = "low"
    
    return {
        "call_id": call_id,
        "empathy_alignment_score": float(empathy_score),
        "alignment_level": alignment_level,
        "component_scores": {
            "speech_rate_alignment": float(speech_rate_alignment * 100),
            "pitch_alignment": float(pitch_alignment * 100),
            "energy_alignment": float(energy_alignment * 100),
        },
        "agent_metrics": {
            "speech_rate": float(agent_speech_rate),
            "pitch_mean": float(agent_pitch),
            "energy_mean": float(agent_energy),
        },
        "customer_metrics": {
            "speech_rate": float(customer_speech_rate),
            "pitch_mean": float(customer_pitch),
            "energy_mean": float(customer_energy),
        },
        "explanation": (
            f"Empathy alignment score: {empathy_score:.1f}/100 ({alignment_level}). "
            f"Speech rate alignment: {speech_rate_alignment*100:.1f}%, "
            f"Pitch alignment: {pitch_alignment*100:.1f}%, "
            f"Energy alignment: {energy_alignment*100:.1f}%."
        ),
    }


def run_behavioral_scoring_pipeline(
    behavior_dir: str | Path = "data/processed/behavior",
    transcripts_dir: str | Path = "data/processed/transcripts",
    diarization_dir: str | Path = "data/processed/diarization",
    output_dir: str | Path = "data/processed/scoring",
) -> List[Dict]:
    """
    Run behavioral scoring pipeline for all calls.
    
    Args:
        behavior_dir: Directory containing behavior CSV files.
        transcripts_dir: Directory containing transcript files.
        diarization_dir: Directory containing diarization JSON files.
        output_dir: Directory to save scoring results.
    
    Returns:
        List of scoring result dictionaries.
    """
    behavior_dir = Path(behavior_dir)
    transcripts_dir = Path(transcripts_dir)
    diarization_dir = Path(diarization_dir)
    output_dir = Path(output_dir)
    
    if not behavior_dir.exists():
        logger.error(f"Behavior directory not found: {behavior_dir}")
        return []
    
    # Find all behavior CSV files
    behavior_files = list(behavior_dir.glob("*.csv"))
    
    if not behavior_files:
        logger.warning(f"No behavior files found in {behavior_dir}")
        return []
    
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - BEHAVIORAL SCORING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Behavior directory: {behavior_dir}")
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Diarization directory: {diarization_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Found {len(behavior_files)} call(s) to score")
    logger.info("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzers
    churn_analyzer = EarlyChurnRiskAnalyzer()
    
    all_results = []
    
    for idx, behavior_file in enumerate(behavior_files, 1):
        call_id = behavior_file.stem
        
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(behavior_files)}] Processing: {call_id}")
        logger.info("-" * 80)
        
        try:
            # Load behavior data
            behavior_df = pd.read_csv(behavior_file)
            
            # Load diarization data
            diarization_file = diarization_dir / f"{call_id}.json"
            if not diarization_file.exists():
                logger.warning(f"Diarization file not found: {diarization_file}")
                continue
            
            with open(diarization_file, "r", encoding="utf-8") as f:
                diarization_data = json.load(f)
            
            # Load transcript
            transcript_file = transcripts_dir / call_id / "transcript.txt"
            if not transcript_file.exists():
                logger.warning(f"Transcript file not found: {transcript_file}")
                continue
            
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            # 1. Calculate stress tendency
            logger.info("Calculating stress tendency...")
            stress_result = calculate_stress_tendency(
                call_id=call_id,
                behavior_df=behavior_df,
                transcript_text=transcript_text,
                diarization_data=diarization_data,
            )
            
            # 2. Calculate empathy alignment
            logger.info("Calculating empathy alignment...")
            empathy_result = calculate_empathy_alignment(
                call_id=call_id,
                behavior_df=behavior_df,
                diarization_data=diarization_data,
            )
            
            # 3. Calculate early churn risk
            logger.info("Calculating early churn risk...")
            # Prepare paths for churn analyzer
            segments_file = transcripts_dir / call_id / "segments.json"
            try:
                churn_result = churn_analyzer.analyze_call(
                    call_id=call_id,
                    audio_features_path=behavior_file,  # Use behavior CSV as audio features
                    diarization_path=diarization_file,
                    transcription_path=segments_file,
                )
            except Exception as e:
                logger.warning(f"Early churn risk calculation failed: {e}")
                churn_result = {
                    "call_id": call_id,
                    "churn_risk_score": 0.0,
                    "risk_level": "low",
                    "risk_factors": {},
                }
            
            # 4. Combine all scores
            combined_result = {
                "call_id": call_id,
                "stress_analysis": stress_result,
                "empathy_analysis": empathy_result,
                "churn_risk_analysis": churn_result,
                "summary": {
                    "stress_tendency": stress_result.get("stress_tendency", "unknown"),
                    "stress_score": stress_result.get("stress_score", 0.0),
                    "empathy_score": empathy_result.get("empathy_alignment_score", 0.0),
                    "churn_risk_level": churn_result.get("risk_level", "unknown"),
                    "churn_risk_score": churn_result.get("churn_risk_score", 0.0),
                },
            }
            
            # Save to JSON
            output_file = output_dir / f"{call_id}_scores.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved scoring results: {output_file}")
            
            # Print summary
            logger.info("")
            logger.info(f"[SUMMARY] Call ID: {call_id}")
            logger.info(f"  Stress tendency: {stress_result.get('stress_tendency', 'N/A')} (score: {stress_result.get('stress_score', 0.0):.2f})")
            logger.info(f"  Empathy alignment: {empathy_result.get('alignment_level', 'N/A')} (score: {empathy_result.get('empathy_alignment_score', 0.0):.1f}/100)")
            logger.info(f"  Churn risk: {churn_result.get('risk_level', 'N/A')} (score: {churn_result.get('churn_risk_score', 0.0):.2f})")
            logger.info(f"  Output file: {output_file}")
            logger.info("")
            
            all_results.append(combined_result)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {call_id}: {e}")
            logger.exception(e)
            continue
    
    # Final summary
    logger.info("=" * 80)
    logger.info("BEHAVIORAL SCORING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total calls processed: {len(all_results)}/{len(behavior_files)}")
    
    if all_results:
        avg_stress = sum(r["summary"]["stress_score"] for r in all_results) / len(all_results)
        avg_empathy = sum(r["summary"]["empathy_score"] for r in all_results) / len(all_results)
        avg_churn = sum(r["summary"]["churn_risk_score"] for r in all_results) / len(all_results)
        
        logger.info(f"Average stress score: {avg_stress:.2f}")
        logger.info(f"Average empathy score: {avg_empathy:.1f}/100")
        logger.info(f"Average churn risk score: {avg_churn:.2f}")
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run behavioral scoring pipeline
    results = run_behavioral_scoring_pipeline(
        behavior_dir="data/processed/behavior",
        transcripts_dir="data/processed/transcripts",
        diarization_dir="data/processed/diarization",
        output_dir="data/processed/scoring",
    )
    
    if results:
        logger.info("\n✓ Behavioral scoring pipeline completed successfully!")
    else:
        logger.warning("\n⚠ No calls were processed.")
