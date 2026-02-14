"""
Call summary report generation module for EdgeMindCX project.

Generates comprehensive call reports combining all analysis results:
- Call summary
- Key risks
- Strengths
- CX score
- Agent recommendations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CallReportGenerator:
    """
    Generates comprehensive call summary reports.
    
    Combines all analysis results into readable JSON and TXT reports.
    """
    
    def __init__(self) -> None:
        """Initialize call report generator."""
        pass
    
    def generate_report(
        self,
        call_id: str,
        cx_score_result: Dict[str, any],
        churn_risk_result: Optional[Dict[str, any]] = None,
        empathy_result: Optional[Dict[str, any]] = None,
        transcription_path: Optional[str | Path] = None,
        call_duration: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Generate comprehensive call report.
        
        Args:
            call_id: Call identifier.
            cx_score_result: Comprehensive CX score result dictionary.
            churn_risk_result: Early churn risk analysis result.
            empathy_result: Empathy convergence analysis result.
            transcription_path: Path to transcription JSON file.
            call_duration: Total call duration in seconds.
        
        Returns:
            Dictionary containing complete call report.
        """
        logger.info(f"Generating call report for: {call_id}")
        
        # Extract key information
        cx_score = cx_score_result.get("cx_score", 0.0)
        component_scores = cx_score_result.get("component_scores", {})
        breakdown = cx_score_result.get("breakdown", {})
        
        # Generate call summary
        call_summary = self._generate_call_summary(
            call_id, cx_score, call_duration, transcription_path
        )
        
        # Identify key risks
        key_risks = self._identify_key_risks(
            cx_score_result, churn_risk_result, empathy_result
        )
        
        # Identify strengths
        strengths = self._identify_strengths(
            cx_score_result, churn_risk_result, empathy_result
        )
        
        # Extract agent recommendations
        agent_recommendations = cx_score_result.get("recommendations", [])
        
        # Add specific recommendations based on risks
        if churn_risk_result:
            churn_signals = churn_risk_result.get("signals", [])
            if churn_signals and churn_risk_result.get("risk_level") in ["medium", "high"]:
                agent_recommendations.append(
                    "URGENT: High churn risk detected. Consider immediate intervention or escalation."
                )
        
        # Create report
        report = {
            "call_id": call_id,
            "report_date": datetime.now().isoformat(),
            "call_summary": call_summary,
            "cx_score": round(cx_score, 2),
            "cx_score_level": self._get_score_level(cx_score),
            "component_scores": component_scores,
            "key_risks": key_risks,
            "strengths": strengths,
            "agent_recommendations": agent_recommendations,
            "detailed_breakdown": breakdown,
        }
        
        logger.info(f"Call report generated for: {call_id}")
        
        return report
    
    def _generate_call_summary(
        self,
        call_id: str,
        cx_score: float,
        call_duration: Optional[float],
        transcription_path: Optional[str | Path],
    ) -> str:
        """
        Generate text summary of the call.
        
        Args:
            call_id: Call identifier.
            cx_score: Overall CX score.
            call_duration: Call duration in seconds.
            transcription_path: Path to transcription file.
        
        Returns:
            Call summary text.
        """
        summary_parts = []
        
        summary_parts.append(f"Call Analysis Report for {call_id}")
        summary_parts.append("=" * 60)
        
        if call_duration:
            minutes = int(call_duration // 60)
            seconds = int(call_duration % 60)
            summary_parts.append(f"Call Duration: {minutes}m {seconds}s")
        
        summary_parts.append(f"Overall CX Score: {cx_score:.1f}/100 ({self._get_score_level(cx_score).upper()})")
        
        # Add transcription preview if available
        if transcription_path:
            transcription_path = Path(transcription_path)
            if transcription_path.exists():
                try:
                    with open(transcription_path, "r", encoding="utf-8") as f:
                        transcription_data = json.load(f)
                    
                    transcript_text = transcription_data.get("text", "")
                    if transcript_text:
                        preview = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
                        summary_parts.append(f"\nCall Transcript Preview:\n{preview}")
                except Exception as e:
                    logger.warning(f"Could not load transcription for summary: {e}")
        
        return "\n".join(summary_parts)
    
    def _identify_key_risks(
        self,
        cx_score_result: Dict[str, any],
        churn_risk_result: Optional[Dict[str, any]],
        empathy_result: Optional[Dict[str, any]],
    ) -> List[str]:
        """
        Identify key risks from analysis results.
        
        Args:
            cx_score_result: CX score analysis result.
            churn_risk_result: Churn risk analysis result.
            empathy_result: Empathy convergence result.
        
        Returns:
            List of key risk strings.
        """
        risks = []
        
        # CX score component risks
        component_scores = cx_score_result.get("component_scores", {})
        
        if component_scores.get("stress", 50.0) < 50:
            risks.append("High stress levels detected - customer may be experiencing frustration")
        
        if component_scores.get("empathy", 50.0) < 60:
            risks.append("Low empathy alignment - agent may not be matching customer's communication style")
        
        if component_scores.get("silence", 50.0) < 50:
            risks.append("Suboptimal silence patterns - conversation pace may be too rushed or too slow")
        
        if component_scores.get("flow", 50.0) < 50:
            risks.append("Poor call flow - turn-taking imbalance or excessive interruptions")
        
        if component_scores.get("churn_risk", 50.0) < 50:
            risks.append("Elevated churn risk - customer shows signs of potential early termination")
        
        # Churn risk specific signals
        if churn_risk_result:
            risk_level = churn_risk_result.get("risk_level", "low")
            if risk_level in ["medium", "high"]:
                signals = churn_risk_result.get("signals", [])
                for signal in signals[:3]:  # Top 3 signals
                    if "No significant" not in signal:
                        risks.append(f"Churn risk: {signal}")
        
        # Empathy convergence risks
        if empathy_result:
            convergence_trend = empathy_result.get("convergence_trend", {})
            trend = convergence_trend.get("trend", "stable")
            if trend == "diverging":
                risks.append("Empathy divergence - communication patterns are moving apart over time")
        
        if not risks:
            risks.append("No significant risks identified")
        
        return risks
    
    def _identify_strengths(
        self,
        cx_score_result: Dict[str, any],
        churn_risk_result: Optional[Dict[str, any]],
        empathy_result: Optional[Dict[str, any]],
    ) -> List[str]:
        """
        Identify strengths from analysis results.
        
        Args:
            cx_score_result: CX score analysis result.
            churn_risk_result: Churn risk analysis result.
            empathy_result: Empathy convergence result.
        
        Returns:
            List of strength strings.
        """
        strengths = []
        
        # CX score component strengths
        component_scores = cx_score_result.get("component_scores", {})
        
        if component_scores.get("stress", 50.0) >= 70:
            strengths.append("Low stress levels - customer appears calm and engaged")
        
        if component_scores.get("empathy", 50.0) >= 75:
            strengths.append("Strong empathy alignment - agent effectively matches customer's communication style")
        
        if component_scores.get("silence", 50.0) >= 75:
            strengths.append("Optimal silence patterns - well-paced conversation")
        
        if component_scores.get("flow", 50.0) >= 70:
            strengths.append("Smooth call flow - balanced turn-taking and natural conversation rhythm")
        
        if component_scores.get("churn_risk", 50.0) >= 70:
            strengths.append("Low churn risk - customer shows positive engagement signals")
        
        # Churn risk strengths
        if churn_risk_result:
            risk_level = churn_risk_result.get("risk_level", "low")
            if risk_level == "low":
                strengths.append("Low early churn risk - customer appears satisfied")
        
        # Empathy convergence strengths
        if empathy_result:
            convergence_trend = empathy_result.get("convergence_trend", {})
            trend = convergence_trend.get("trend", "stable")
            if trend == "converging":
                strengths.append("Positive empathy convergence - communication patterns improving over time")
            
            empathy_score = empathy_result.get("empathy_alignment_score", 0.0)
            if empathy_score >= 75:
                strengths.append("Excellent empathy alignment - strong rapport building")
        
        if not strengths:
            strengths.append("Standard performance across all metrics")
        
        return strengths
    
    def _get_score_level(self, score: float) -> str:
        """
        Get score level description.
        
        Args:
            score: Score value (0-100).
        
        Returns:
            Score level string.
        """
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 50:
            return "moderate"
        else:
            return "poor"
    
    def save_report(
        self,
        report: Dict[str, any],
        output_dir: str | Path,
        formats: List[str] = ["json", "txt"],
    ) -> Dict[str, Path]:
        """
        Save report in multiple formats.
        
        Args:
            report: Report dictionary.
            output_dir: Output directory.
            formats: List of formats to save ("json", "txt"). Default is both.
        
        Returns:
            Dictionary with paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        call_id = report["call_id"]
        saved_paths = {}
        
        # Save JSON
        if "json" in formats:
            json_path = output_dir / f"call_report_{call_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            saved_paths["json"] = json_path
            logger.info(f"Saved JSON report: {json_path}")
        
        # Save TXT
        if "txt" in formats:
            txt_path = output_dir / f"call_report_{call_id}.txt"
            txt_content = self._format_txt_report(report)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)
            saved_paths["txt"] = txt_path
            logger.info(f"Saved TXT report: {txt_path}")
        
        return saved_paths
    
    def _format_txt_report(self, report: Dict[str, any]) -> str:
        """
        Format report as readable text.
        
        Args:
            report: Report dictionary.
        
        Returns:
            Formatted text string.
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("EDGEMINDCX - CALL ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Call Summary
        lines.append("CALL SUMMARY")
        lines.append("-" * 80)
        lines.append(report.get("call_summary", ""))
        lines.append("")
        
        # CX Score
        lines.append("CUSTOMER EXPERIENCE SCORE")
        lines.append("-" * 80)
        cx_score = report.get("cx_score", 0.0)
        score_level = report.get("cx_score_level", "unknown")
        lines.append(f"Overall Score: {cx_score:.1f}/100 ({score_level.upper()})")
        lines.append("")
        
        # Component Scores
        component_scores = report.get("component_scores", {})
        lines.append("Component Breakdown:")
        for component, score in component_scores.items():
            lines.append(f"  - {component.replace('_', ' ').title()}: {score:.1f}/100")
        lines.append("")
        
        # Key Risks
        lines.append("KEY RISKS")
        lines.append("-" * 80)
        key_risks = report.get("key_risks", [])
        for i, risk in enumerate(key_risks, 1):
            lines.append(f"{i}. {risk}")
        lines.append("")
        
        # Strengths
        lines.append("STRENGTHS")
        lines.append("-" * 80)
        strengths = report.get("strengths", [])
        for i, strength in enumerate(strengths, 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
        
        # Agent Recommendations
        lines.append("AGENT RECOMMENDATIONS")
        lines.append("-" * 80)
        recommendations = report.get("agent_recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append(f"Report Generated: {report.get('report_date', 'N/A')}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def generate_call_report(
    call_id: str,
    cx_score_result: Dict[str, any],
    churn_risk_result: Optional[Dict[str, any]] = None,
    empathy_result: Optional[Dict[str, any]] = None,
    transcription_path: Optional[str | Path] = None,
    call_duration: Optional[float] = None,
    output_dir: Optional[str | Path] = None,
    save_reports: bool = True,
) -> Dict[str, any]:
    """
    Convenience function to generate call report.
    
    Args:
        call_id: Call identifier.
        cx_score_result: Comprehensive CX score result.
        churn_risk_result: Early churn risk analysis result.
        empathy_result: Empathy convergence analysis result.
        transcription_path: Path to transcription JSON file.
        call_duration: Total call duration in seconds.
        output_dir: Output directory for saving reports.
        save_reports: Whether to save reports to files. Default is True.
    
    Returns:
        Dictionary containing complete call report.
    
    Example:
        >>> report = generate_call_report(
        ...     call_id="call_abc123",
        ...     cx_score_result=cx_result,
        ...     churn_risk_result=churn_result,
        ...     empathy_result=empathy_result
        ... )
    """
    generator = CallReportGenerator()
    
    report = generator.generate_report(
        call_id=call_id,
        cx_score_result=cx_score_result,
        churn_risk_result=churn_risk_result,
        empathy_result=empathy_result,
        transcription_path=transcription_path,
        call_duration=call_duration,
    )
    
    if save_reports and output_dir:
        generator.save_report(
            report=report,
            output_dir=output_dir,
            formats=["json", "txt"],
        )
    
    return report
