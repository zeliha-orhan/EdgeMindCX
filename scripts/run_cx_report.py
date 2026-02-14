"""
Script to generate final CX reports for each call.

Generates:
1. CX Score (0-100)
2. Top 3 critical problems
3. Agent action recommendations
4. Short call summary

Output: JSON and TXT files
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

# Import modules directly
import importlib.util

# ComprehensiveCXScore
spec_cx = importlib.util.spec_from_file_location(
    "comprehensive_cx_score_module",
    project_root / "edge_mind_cx" / "scoring" / "comprehensive_cx_score.py"
)
cx_score_module = importlib.util.module_from_spec(spec_cx)
spec_cx.loader.exec_module(cx_score_module)
ComprehensiveCXScore = cx_score_module.ComprehensiveCXScore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def identify_top_problems(
    scoring_data: Dict,
    behavior_df: pd.DataFrame,
) -> List[Dict[str, any]]:
    """
    Identify top 3 critical problems from analysis results.
    
    Args:
        scoring_data: Scoring results dictionary.
        behavior_df: Behavior metrics DataFrame.
    
    Returns:
        List of top 3 problems with descriptions and severity.
    """
    problems = []
    
    # Problem 1: High churn risk
    churn_risk = scoring_data.get("churn_risk_analysis", {})
    risk_level = churn_risk.get("risk_level", "low")
    risk_score = churn_risk.get("risk_score", 0.0)
    
    if risk_level in ["medium", "high"]:
        signals = churn_risk.get("signals", [])
        problem_desc = f"Early churn risk detected ({risk_level.upper()})"
        if signals:
            problem_desc += f": {signals[0]}"
        problems.append({
            "severity": risk_level,
            "category": "Churn Risk",
            "description": problem_desc,
            "score": risk_score,
            "recommendation": "Monitor customer closely and consider proactive intervention."
        })
    
    # Problem 2: Long silences
    long_silences = churn_risk.get("risk_factors", {}).get("long_silences", {})
    if long_silences.get("risk_score", 0.0) > 50:
        num_silences = long_silences.get("num_long_silences", 0)
        max_silence = long_silences.get("max_silence", 0.0)
        problems.append({
            "severity": "high" if num_silences > 10 else "medium",
            "category": "Call Flow",
            "description": f"Multiple long silences detected ({num_silences} instances, max: {max_silence:.1f}s)",
            "score": long_silences.get("risk_score", 0.0),
            "recommendation": "Improve response time and reduce awkward pauses. Consider active listening techniques."
        })
    
    # Problem 3: Low empathy alignment
    empathy_analysis = scoring_data.get("empathy_analysis", {})
    empathy_score = empathy_analysis.get("empathy_alignment_score", 100.0)
    if empathy_score < 70:
        problems.append({
            "severity": "medium" if empathy_score < 50 else "low",
            "category": "Empathy",
            "description": f"Low empathy alignment ({empathy_score:.1f}/100)",
            "score": 100.0 - empathy_score,
            "recommendation": "Work on matching customer's communication style (speech rate, tone, energy)."
        })
    
    # Problem 4: High stress
    stress_analysis = scoring_data.get("stress_analysis", {})
    stress_score = stress_analysis.get("stress_score", 0.0)
    if stress_score > 0.6:
        problems.append({
            "severity": "high" if stress_score > 0.8 else "medium",
            "category": "Customer Stress",
            "description": f"High customer stress detected (score: {stress_score:.2f})",
            "score": stress_score * 100,
            "recommendation": "Focus on de-escalation and empathy. Acknowledge customer concerns."
        })
    
    # Problem 5: Negative word density
    negative_words = churn_risk.get("risk_factors", {}).get("negative_word_density", {})
    if negative_words.get("risk_score", 0.0) > 50:
        density = negative_words.get("negative_word_density", 0.0)
        problems.append({
            "severity": "high" if density > 0.2 else "medium",
            "category": "Sentiment",
            "description": f"High negative word density ({density*100:.1f}%)",
            "score": negative_words.get("risk_score", 0.0),
            "recommendation": "Address negative sentiment proactively. Use positive language and solutions."
        })
    
    # Sort by severity and score, take top 3
    severity_order = {"high": 3, "medium": 2, "low": 1}
    problems.sort(
        key=lambda x: (severity_order.get(x["severity"], 0), x["score"]),
        reverse=True
    )
    
    return problems[:3]


def generate_agent_recommendations(
    problems: List[Dict],
    scoring_data: Dict,
    cx_score: float,
) -> List[str]:
    """
    Generate actionable recommendations for the agent.
    
    Args:
        problems: List of identified problems.
        scoring_data: Scoring results dictionary.
        cx_score: Overall CX score.
    
    Returns:
        List of recommendation strings.
    """
    recommendations = []
    
    # General recommendation based on CX score
    if cx_score < 50:
        recommendations.append("URGENT: Overall CX score is low. Immediate attention required.")
    elif cx_score < 70:
        recommendations.append("MODERATE: CX score indicates room for improvement.")
    else:
        recommendations.append("GOOD: Maintain current performance level.")
    
    # Problem-specific recommendations
    for problem in problems:
        rec = problem.get("recommendation", "")
        if rec:
            recommendations.append(f"[{problem['category']}] {rec}")
    
    # Empathy-specific recommendations
    empathy_analysis = scoring_data.get("empathy_analysis", {})
    empathy_score = empathy_analysis.get("empathy_alignment_score", 100.0)
    if empathy_score < 80:
        component_scores = empathy_analysis.get("component_scores", {})
        if component_scores.get("speech_rate_alignment", 100) < 80:
            recommendations.append(
                "[Empathy] Adjust speech rate to better match customer's pace."
            )
        if component_scores.get("pitch_alignment", 100) < 80:
            recommendations.append(
                "[Empathy] Match customer's tone and pitch for better rapport."
            )
    
    # Stress-specific recommendations
    stress_analysis = scoring_data.get("stress_analysis", {})
    stress_score = stress_analysis.get("stress_score", 0.0)
    if stress_score > 0.5:
        recommendations.append(
            "[Stress Management] Customer shows signs of stress. Use calming language and active listening."
        )
    
    return recommendations


def generate_call_summary(
    call_id: str,
    cx_score: float,
    transcript_text: str,
    call_duration: Optional[float] = None,
) -> str:
    """
    Generate a short call summary.
    
    Args:
        call_id: Call identifier.
        cx_score: Overall CX score.
        transcript_text: Full transcript text.
        call_duration: Call duration in seconds.
    
    Returns:
        Short call summary text.
    """
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"Call ID: {call_id}")
    if call_duration:
        minutes = int(call_duration // 60)
        seconds = int(call_duration % 60)
        summary_parts.append(f"Duration: {minutes}m {seconds}s")
    
    summary_parts.append(f"CX Score: {cx_score:.1f}/100")
    
    # Score interpretation
    if cx_score >= 80:
        score_level = "Excellent"
    elif cx_score >= 70:
        score_level = "Good"
    elif cx_score >= 60:
        score_level = "Fair"
    elif cx_score >= 50:
        score_level = "Poor"
    else:
        score_level = "Critical"
    
    summary_parts.append(f"Performance Level: {score_level}")
    
    # Transcript preview (first 300 chars)
    if transcript_text:
        preview = transcript_text[:300].strip()
        if len(transcript_text) > 300:
            preview += "..."
        summary_parts.append(f"\nTranscript Preview:\n{preview}")
    
    return "\n".join(summary_parts)


def generate_cx_report(
    call_id: str,
    scoring_data: Dict,
    behavior_df: pd.DataFrame,
    transcript_text: str,
    call_duration: Optional[float] = None,
) -> Dict[str, any]:
    """
    Generate comprehensive CX report for a call.
    
    Args:
        call_id: Call identifier.
        scoring_data: Scoring results dictionary.
        behavior_df: Behavior metrics DataFrame.
        transcript_text: Full transcript text.
        call_duration: Call duration in seconds.
    
    Returns:
        Complete CX report dictionary.
    """
    logger.info(f"Generating CX report for {call_id}...")
    
    # Calculate CX score using ComprehensiveCXScore
    cx_calculator = ComprehensiveCXScore()
    
    # Extract required data
    stress_score = scoring_data.get("stress_analysis", {}).get("stress_score", 0.0)
    empathy_score = scoring_data.get("empathy_analysis", {}).get("empathy_alignment_score", 0.0)
    churn_risk_score = scoring_data.get("churn_risk_analysis", {}).get("risk_score", 0.0)
    
    # Calculate silence metrics from behavior data
    silence_metrics = {}
    if not behavior_df.empty:
        silence_rows = behavior_df[behavior_df["metric"] == "total_silence_seconds"]
        speaking_rows = behavior_df[behavior_df["metric"] == "total_speaking_time_seconds"]
        
        if not silence_rows.empty and not speaking_rows.empty:
            total_silence = silence_rows["value"].sum()
            total_speaking = speaking_rows["value"].sum()
            total_duration = total_silence + total_speaking
            
            if total_duration > 0:
                silence_metrics = {
                    "silence_ratio": float(total_silence / total_duration),
                    "silence_duration": float(total_silence),
                    "total_duration": float(total_duration),
                }
    
    # Calculate CX score
    cx_result = cx_calculator.calculate_cx_score(
        call_id=call_id,
        stress_score=stress_score,
        empathy_alignment_score=empathy_score,
        silence_metrics=silence_metrics if silence_metrics else None,
        speaker_dynamics_path=None,  # Can be added if needed
        churn_risk_score=churn_risk_score,
    )
    
    cx_score = cx_result.get("cx_score", 0.0)
    
    # Identify top 3 problems
    top_problems = identify_top_problems(scoring_data, behavior_df)
    
    # Generate agent recommendations
    agent_recommendations = generate_agent_recommendations(
        top_problems, scoring_data, cx_score
    )
    
    # Generate call summary
    call_summary = generate_call_summary(
        call_id, cx_score, transcript_text, call_duration
    )
    
    # Create report
    report = {
        "call_id": call_id,
        "cx_score": round(cx_score, 2),
        "cx_score_level": "excellent" if cx_score >= 80 else "good" if cx_score >= 70 else "fair" if cx_score >= 60 else "poor" if cx_score >= 50 else "critical",
        "top_3_problems": top_problems,
        "agent_recommendations": agent_recommendations,
        "call_summary": call_summary,
        "component_scores": {
            "stress": round(stress_score, 3),
            "empathy": round(empathy_score, 2),
            "churn_risk": round(churn_risk_score, 2),
        },
        "detailed_breakdown": cx_result.get("breakdown", {}),
    }
    
    return report


def generate_text_report(report_data: Dict[str, any]) -> str:
    """
    Generate human-readable text report from JSON data.
    
    Args:
        report_data: Report dictionary.
    
    Returns:
        Formatted text report.
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("EDGEMINDCX - CUSTOMER EXPERIENCE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Call ID and CX Score
    lines.append(f"Call ID: {report_data['call_id']}")
    lines.append(f"CX Score: {report_data['cx_score']}/100 ({report_data['cx_score_level'].upper()})")
    lines.append("")
    
    # Call Summary
    lines.append("-" * 80)
    lines.append("CALL SUMMARY")
    lines.append("-" * 80)
    lines.append(report_data.get("call_summary", "N/A"))
    lines.append("")
    
    # Top 3 Problems
    lines.append("-" * 80)
    lines.append("TOP 3 CRITICAL PROBLEMS")
    lines.append("-" * 80)
    problems = report_data.get("top_3_problems", [])
    if problems:
        for i, problem in enumerate(problems, 1):
            lines.append(f"\n{i}. [{problem['severity'].upper()}] {problem['category']}")
            lines.append(f"   Description: {problem['description']}")
            lines.append(f"   Score: {problem['score']:.1f}")
            lines.append(f"   Recommendation: {problem['recommendation']}")
    else:
        lines.append("No critical problems identified.")
    lines.append("")
    
    # Agent Recommendations
    lines.append("-" * 80)
    lines.append("AGENT ACTION RECOMMENDATIONS")
    lines.append("-" * 80)
    recommendations = report_data.get("agent_recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
    else:
        lines.append("No specific recommendations.")
    lines.append("")
    
    # Component Scores
    lines.append("-" * 80)
    lines.append("COMPONENT SCORES")
    lines.append("-" * 80)
    component_scores = report_data.get("component_scores", {})
    lines.append(f"Stress Score: {component_scores.get('stress', 0.0):.3f}")
    lines.append(f"Empathy Score: {component_scores.get('empathy', 0.0):.2f}/100")
    lines.append(f"Churn Risk Score: {component_scores.get('churn_risk', 0.0):.2f}/100")
    lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def run_cx_report_pipeline(
    scoring_dir: str | Path = "data/processed/scoring",
    behavior_dir: str | Path = "data/processed/behavior",
    transcripts_dir: str | Path = "data/processed/transcripts",
    diarization_dir: str | Path = "data/processed/diarization",
    output_dir: str | Path = "data/processed/reports",
) -> List[Dict]:
    """
    Run CX report generation pipeline for all calls.
    
    Args:
        scoring_dir: Directory containing scoring JSON files.
        behavior_dir: Directory containing behavior CSV files.
        transcripts_dir: Directory containing transcript files.
        diarization_dir: Directory containing diarization JSON files.
        output_dir: Directory to save reports.
    
    Returns:
        List of report dictionaries.
    """
    scoring_dir = Path(scoring_dir)
    behavior_dir = Path(behavior_dir)
    transcripts_dir = Path(transcripts_dir)
    output_dir = Path(output_dir)
    
    if not scoring_dir.exists():
        logger.error(f"Scoring directory not found: {scoring_dir}")
        return []
    
    # Find all scoring JSON files
    scoring_files = list(scoring_dir.glob("*_scores.json"))
    
    if not scoring_files:
        logger.warning(f"No scoring files found in {scoring_dir}")
        return []
    
    logger.info("=" * 80)
    logger.info("EDGEMINDCX - CX REPORT GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Scoring directory: {scoring_dir}")
    logger.info(f"Behavior directory: {behavior_dir}")
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Found {len(scoring_files)} call(s) to process")
    logger.info("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_reports = []
    
    for idx, scoring_file in enumerate(scoring_files, 1):
        call_id = scoring_file.stem.replace("_scores", "")
        
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(scoring_files)}] Processing: {call_id}")
        logger.info("-" * 80)
        
        try:
            # Load scoring data
            with open(scoring_file, "r", encoding="utf-8") as f:
                scoring_data = json.load(f)
            
            # Load behavior data
            behavior_file = behavior_dir / f"{call_id}.csv"
            if not behavior_file.exists():
                logger.warning(f"Behavior file not found: {behavior_file}")
                continue
            
            behavior_df = pd.read_csv(behavior_file)
            
            # Load transcript
            transcript_file = transcripts_dir / call_id / "transcript.txt"
            if not transcript_file.exists():
                logger.warning(f"Transcript file not found: {transcript_file}")
                continue
            
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            # Get call duration from diarization if available
            diarization_file = Path(diarization_dir) / f"{call_id}.json"
            call_duration = None
            if diarization_file.exists():
                with open(diarization_file, "r", encoding="utf-8") as f:
                    diarization_data = json.load(f)
                call_duration = diarization_data.get("total_call_duration")
            
            # Generate report
            report = generate_cx_report(
                call_id=call_id,
                scoring_data=scoring_data,
                behavior_df=behavior_df,
                transcript_text=transcript_text,
                call_duration=call_duration,
            )
            
            # Save JSON report
            json_output = output_dir / f"cx_report_{call_id}.json"
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Generate and save TXT report
            text_report = generate_text_report(report)
            txt_output = output_dir / f"cx_report_{call_id}.txt"
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(text_report)
            
            logger.info(f"Saved JSON report: {json_output}")
            logger.info(f"Saved TXT report: {txt_output}")
            
            # Print summary
            logger.info("")
            logger.info(f"[SUMMARY] Call ID: {call_id}")
            logger.info(f"  CX Score: {report['cx_score']}/100 ({report['cx_score_level'].upper()})")
            logger.info(f"  Top Problems: {len(report['top_3_problems'])}")
            logger.info(f"  Recommendations: {len(report['agent_recommendations'])}")
            logger.info("")
            
            all_reports.append(report)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to process {call_id}: {e}")
            logger.exception(e)
            continue
    
    # Final summary
    logger.info("=" * 80)
    logger.info("CX REPORT GENERATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total reports generated: {len(all_reports)}/{len(scoring_files)}")
    
    if all_reports:
        avg_cx = sum(r["cx_score"] for r in all_reports) / len(all_reports)
        logger.info(f"Average CX Score: {avg_cx:.2f}/100")
    
    logger.info("=" * 80)
    
    return all_reports


if __name__ == "__main__":
    # Run CX report generation pipeline
    reports = run_cx_report_pipeline(
        scoring_dir="data/processed/scoring",
        behavior_dir="data/processed/behavior",
        transcripts_dir="data/processed/transcripts",
        diarization_dir="data/processed/diarization",
        output_dir="data/processed/reports",
    )
    
    if reports:
        logger.info("\n✓ CX report generation completed successfully!")
    else:
        logger.warning("\n⚠ No reports were generated.")
