"""
Comprehensive CX Score calculation module for EdgeMindCX project.

Combines all analysis components to produce an explainable CX score (0-100):
- Stress score
- Empathy alignment
- Silence quality
- Call flow organization
- Early churn risk
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ComprehensiveCXScore:
    """
    Calculates comprehensive CX score from all analysis components.
    
    Integrates:
    - Stress score (inverted: lower stress = higher CX)
    - Empathy alignment (higher empathy = higher CX)
    - Silence quality (optimal silence = higher CX)
    - Call flow organization (smooth flow = higher CX)
    - Early churn risk (inverted: lower risk = higher CX)
    """
    
    def __init__(
        self,
        stress_weight: float = 0.25,
        empathy_weight: float = 0.25,
        silence_weight: float = 0.15,
        flow_weight: float = 0.15,
        churn_risk_weight: float = 0.20,
        normalize_weights: bool = True,
    ) -> None:
        """
        Initialize comprehensive CX score calculator.
        
        Args:
            stress_weight: Weight for stress component (inverted). Default is 0.25.
            empathy_weight: Weight for empathy component. Default is 0.25.
            silence_weight: Weight for silence quality component. Default is 0.15.
            flow_weight: Weight for call flow organization component. Default is 0.15.
            churn_risk_weight: Weight for early churn risk component (inverted).
                             Default is 0.20.
            normalize_weights: If True, automatically normalize weights to sum to 1.0.
                             Default is True.
        """
        self.stress_weight = stress_weight
        self.empathy_weight = empathy_weight
        self.silence_weight = silence_weight
        self.flow_weight = flow_weight
        self.churn_risk_weight = churn_risk_weight
        
        # Validate and normalize weights
        if normalize_weights:
            total = (
                stress_weight
                + empathy_weight
                + silence_weight
                + flow_weight
                + churn_risk_weight
            )
            if total > 0:
                self.stress_weight = stress_weight / total
                self.empathy_weight = empathy_weight / total
                self.silence_weight = silence_weight / total
                self.flow_weight = flow_weight / total
                self.churn_risk_weight = churn_risk_weight / total
                logger.info(
                    f"Weights normalized. Final weights: "
                    f"stress={self.stress_weight:.2f}, "
                    f"empathy={self.empathy_weight:.2f}, "
                    f"silence={self.silence_weight:.2f}, "
                    f"flow={self.flow_weight:.2f}, "
                    f"churn_risk={self.churn_risk_weight:.2f}"
                )
            else:
                raise ValueError("Sum of weights must be greater than 0")
        else:
            total = (
                stress_weight
                + empathy_weight
                + silence_weight
                + flow_weight
                + churn_risk_weight
            )
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0 (current sum: {total}). "
                    f"Set normalize_weights=True to auto-normalize."
                )
        
        logger.info("ComprehensiveCXScore initialized")
    
    def calculate_cx_score(
        self,
        call_id: str,
        stress_score: Optional[float] = None,
        empathy_alignment_score: Optional[float] = None,
        silence_metrics: Optional[Dict[str, float]] = None,
        speaker_dynamics_path: Optional[str | Path] = None,
        churn_risk_score: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Calculate comprehensive CX score from all analysis components.
        
        Args:
            call_id: Call identifier.
            stress_score: Unified stress score (0-1). Higher = more stress.
            empathy_alignment_score: Empathy alignment score (0-100).
            silence_metrics: Dictionary with silence metrics.
                           Expected keys: silence_ratio, silence_duration, total_duration.
            speaker_dynamics_path: Path to speaker dynamics CSV file.
                                 Used for call flow analysis.
            churn_risk_score: Early churn risk score (0-100). Higher = more risk.
        
        Returns:
            Dictionary containing:
            - call_id: Call identifier
            - cx_score: Overall CX score (0-100)
            - component_scores: Individual component scores
            - breakdown: Detailed breakdown of all components
            - explanation: Human-readable explanation
            - recommendations: Actionable recommendations
        """
        logger.info(f"Calculating comprehensive CX score for call: {call_id}")
        
        # Calculate individual component scores
        stress_component = self._calculate_stress_component(stress_score)
        empathy_component = self._calculate_empathy_component(empathy_alignment_score)
        silence_component = self._calculate_silence_component(silence_metrics)
        flow_component = self._calculate_flow_component(speaker_dynamics_path)
        churn_risk_component = self._calculate_churn_risk_component(churn_risk_score)
        
        # Weighted combination
        cx_score = (
            self.stress_weight * stress_component
            + self.empathy_weight * empathy_component
            + self.silence_weight * silence_component
            + self.flow_weight * flow_component
            + self.churn_risk_weight * churn_risk_component
        )
        
        # Clamp to [0, 100]
        cx_score = max(0.0, min(100.0, cx_score))
        
        # Create breakdown
        breakdown = {
            "stress": {
                "raw_score": stress_score if stress_score is not None else 0.5,
                "component_score": stress_component,
                "weight": self.stress_weight,
                "contribution": self.stress_weight * stress_component,
                "description": "Lower stress indicates better CX",
            },
            "empathy": {
                "raw_score": empathy_alignment_score if empathy_alignment_score is not None else 50.0,
                "component_score": empathy_component,
                "weight": self.empathy_weight,
                "contribution": self.empathy_weight * empathy_component,
                "description": "Higher empathy alignment indicates better CX",
            },
            "silence": {
                "component_score": silence_component,
                "weight": self.silence_weight,
                "contribution": self.silence_weight * silence_component,
                "description": "Optimal silence range (10-30%) indicates better CX",
            },
            "flow": {
                "component_score": flow_component,
                "weight": self.flow_weight,
                "contribution": self.flow_weight * flow_component,
                "description": "Smooth call flow with balanced turn-taking indicates better CX",
            },
            "churn_risk": {
                "raw_score": churn_risk_score if churn_risk_score is not None else 0.0,
                "component_score": churn_risk_component,
                "weight": self.churn_risk_weight,
                "contribution": self.churn_risk_weight * churn_risk_component,
                "description": "Lower churn risk indicates better CX",
            },
        }
        
        # Generate explanation
        explanation = self._generate_explanation(cx_score, breakdown)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(breakdown)
        
        result = {
            "call_id": call_id,
            "cx_score": round(cx_score, 2),
            "component_scores": {
                "stress": round(stress_component, 2),
                "empathy": round(empathy_component, 2),
                "silence": round(silence_component, 2),
                "flow": round(flow_component, 2),
                "churn_risk": round(churn_risk_component, 2),
            },
            "breakdown": breakdown,
            "explanation": explanation,
            "recommendations": recommendations,
        }
        
        logger.info(f"CX Score calculated: {cx_score:.2f}/100")
        
        return result
    
    def _calculate_stress_component(
        self,
        stress_score: Optional[float],
    ) -> float:
        """
        Calculate stress component for CX score.
        
        Stress is inverted: lower stress = higher CX.
        
        Args:
            stress_score: Stress score (0-1). Higher = more stress.
        
        Returns:
            Stress component score (0-100).
        """
        if stress_score is None:
            logger.warning("No stress score provided, using neutral 50.0")
            return 50.0
        
        # Invert: lower stress = higher CX
        stress_component = (1.0 - np.clip(stress_score, 0.0, 1.0)) * 100.0
        
        return float(stress_component)
    
    def _calculate_empathy_component(
        self,
        empathy_alignment_score: Optional[float],
    ) -> float:
        """
        Calculate empathy component for CX score.
        
        Args:
            empathy_alignment_score: Empathy alignment score (0-100).
        
        Returns:
            Empathy component score (0-100).
        """
        if empathy_alignment_score is None:
            logger.warning("No empathy alignment score provided, using neutral 50.0")
            return 50.0
        
        # Direct mapping: already 0-100
        empathy_component = np.clip(empathy_alignment_score, 0.0, 100.0)
        
        return float(empathy_component)
    
    def _calculate_silence_component(
        self,
        silence_metrics: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate silence quality component for CX score.
        
        Optimal silence range: 10-30% of call duration.
        
        Args:
            silence_metrics: Dictionary with silence metrics.
        
        Returns:
            Silence component score (0-100).
        """
        if silence_metrics is None:
            logger.warning("No silence metrics provided, using neutral 50.0")
            return 50.0
        
        silence_ratio = silence_metrics.get("silence_ratio", 0.0)
        silence_ratio = np.clip(silence_ratio, 0.0, 1.0)
        
        # Optimal range: 10-30%
        OPTIMAL_MIN = 0.1
        OPTIMAL_MAX = 0.3
        
        if OPTIMAL_MIN <= silence_ratio <= OPTIMAL_MAX:
            # Perfect range
            score = 100.0
        elif silence_ratio < OPTIMAL_MIN:
            # Too little silence
            score = 50.0 + (silence_ratio / OPTIMAL_MIN) * 50.0
        else:
            # Too much silence
            excess = silence_ratio - OPTIMAL_MAX
            max_excess = 1.0 - OPTIMAL_MAX
            score = max(0.0, 100.0 - (excess / max_excess) * 100.0)
        
        return float(score)
    
    def _calculate_flow_component(
        self,
        speaker_dynamics_path: Optional[str | Path],
    ) -> float:
        """
        Calculate call flow organization component for CX score.
        
        Analyzes:
        - Turn-taking balance
        - Overlap frequency (some overlap is good, too much is bad)
        - Conversation rhythm
        
        Args:
            speaker_dynamics_path: Path to speaker dynamics CSV file.
        
        Returns:
            Flow component score (0-100).
        """
        if speaker_dynamics_path is None:
            logger.warning("No speaker dynamics provided, using neutral 50.0")
            return 50.0
        
        speaker_dynamics_path = Path(speaker_dynamics_path)
        
        if not speaker_dynamics_path.exists():
            logger.warning(f"Speaker dynamics file not found: {speaker_dynamics_path}")
            return 50.0
        
        try:
            df = pd.read_csv(speaker_dynamics_path)
            
            # Get overlap metrics
            overlap_data = df[df["speaker"] == "overlap"]
            
            if len(overlap_data) == 0:
                # No overlap data
                return 50.0
            
            # Get total overlap and number of overlaps
            total_overlap = overlap_data[
                overlap_data["metric"] == "total_overlap_seconds"
            ]["value"].values
            
            num_overlaps = overlap_data[
                overlap_data["metric"] == "num_overlaps"
            ]["value"].values
            
            if len(total_overlap) == 0 or len(num_overlaps) == 0:
                return 50.0
            
            total_overlap = total_overlap[0]
            num_overlaps = num_overlaps[0]
            
            # Get agent and customer speaking times
            agent_data = df[df["speaker"] == "agent"]
            customer_data = df[df["speaker"] == "customer"]
            
            agent_speaking = agent_data[
                agent_data["metric"] == "total_speaking_time_seconds"
            ]["value"].values
            
            customer_speaking = customer_data[
                customer_data["metric"] == "total_speaking_time_seconds"
            ]["value"].values
            
            # Calculate flow score
            flow_score = 50.0  # Base score
            
            # Turn-taking balance (50-50 is ideal)
            if len(agent_speaking) > 0 and len(customer_speaking) > 0:
                total_time = agent_speaking[0] + customer_speaking[0]
                if total_time > 0:
                    agent_ratio = agent_speaking[0] / total_time
                    balance_score = 1.0 - abs(agent_ratio - 0.5) * 2.0  # 0.5 = perfect balance
                    flow_score += balance_score * 30.0
            
            # Overlap analysis (moderate overlap is good, excessive is bad)
            if total_overlap > 0:
                # Optimal: 2-5% of total call time
                overlap_ratio = total_overlap / max(total_time, 1.0) if total_time > 0 else 0.0
                
                if 0.02 <= overlap_ratio <= 0.05:
                    # Optimal overlap
                    flow_score += 20.0
                elif overlap_ratio < 0.02:
                    # Too little overlap (stiff conversation)
                    flow_score += overlap_ratio / 0.02 * 10.0
                else:
                    # Too much overlap (interruptions)
                    excess = overlap_ratio - 0.05
                    flow_score -= min(20.0, excess * 200.0)
            
            # Number of overlaps (moderate is good)
            if num_overlaps > 0:
                # Optimal: 3-8 overlaps per call
                if 3 <= num_overlaps <= 8:
                    flow_score += 10.0
                elif num_overlaps < 3:
                    flow_score += (num_overlaps / 3.0) * 5.0
                else:
                    # Too many overlaps
                    flow_score -= min(10.0, (num_overlaps - 8) * 1.0)
            
            return float(np.clip(flow_score, 0.0, 100.0))
            
        except Exception as e:
            logger.error(f"Error calculating flow component: {e}")
            return 50.0
    
    def _calculate_churn_risk_component(
        self,
        churn_risk_score: Optional[float],
    ) -> float:
        """
        Calculate churn risk component for CX score.
        
        Churn risk is inverted: lower risk = higher CX.
        
        Args:
            churn_risk_score: Early churn risk score (0-100). Higher = more risk.
        
        Returns:
            Churn risk component score (0-100).
        """
        if churn_risk_score is None:
            logger.warning("No churn risk score provided, using neutral 50.0")
            return 50.0
        
        # Invert: lower risk = higher CX
        churn_risk_component = 100.0 - np.clip(churn_risk_score, 0.0, 100.0)
        
        return float(churn_risk_component)
    
    def _generate_explanation(
        self,
        cx_score: float,
        breakdown: Dict[str, Dict[str, any]],
    ) -> str:
        """
        Generate human-readable explanation of CX score.
        
        Args:
            cx_score: Overall CX score (0-100).
            breakdown: Component breakdown dictionary.
        
        Returns:
            Explanation string.
        """
        # Determine overall level
        if cx_score >= 80:
            level = "excellent"
            base_text = "Excellent customer experience"
        elif cx_score >= 65:
            level = "good"
            base_text = "Good customer experience"
        elif cx_score >= 50:
            level = "moderate"
            base_text = "Moderate customer experience"
        else:
            level = "poor"
            base_text = "Poor customer experience"
        
        explanation = f"{base_text} (Score: {cx_score:.1f}/100). "
        
        # Identify strongest and weakest components
        components = {
            "stress": breakdown["stress"]["component_score"],
            "empathy": breakdown["empathy"]["component_score"],
            "silence": breakdown["silence"]["component_score"],
            "flow": breakdown["flow"]["component_score"],
            "churn_risk": breakdown["churn_risk"]["component_score"],
        }
        
        strongest = max(components.items(), key=lambda x: x[1])
        weakest = min(components.items(), key=lambda x: x[1])
        
        # Add component insights
        if strongest[1] >= 75:
            explanation += f"Strong {strongest[0]} performance. "
        
        if weakest[1] < 50:
            explanation += f"Opportunity to improve {weakest[0]} (current: {weakest[1]:.1f}/100)."
        else:
            explanation += "All components show reasonable performance."
        
        return explanation.strip()
    
    def _generate_recommendations(
        self,
        breakdown: Dict[str, Dict[str, any]],
    ) -> List[str]:
        """
        Generate actionable recommendations based on component scores.
        
        Args:
            breakdown: Component breakdown dictionary.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Stress recommendations
        if breakdown["stress"]["component_score"] < 50:
            recommendations.append(
                "Reduce stress: Agent should use calmer tone and slower pace to help customer relax."
            )
        
        # Empathy recommendations
        if breakdown["empathy"]["component_score"] < 60:
            recommendations.append(
                "Improve empathy: Agent should better match customer's communication style and emotional state."
            )
        
        # Silence recommendations
        silence_score = breakdown["silence"]["component_score"]
        if silence_score < 50:
            recommendations.append(
                "Optimize silence: Balance conversation pace - avoid rushed or overly slow interactions."
            )
        
        # Flow recommendations
        if breakdown["flow"]["component_score"] < 50:
            recommendations.append(
                "Improve call flow: Better turn-taking balance and reduce excessive interruptions."
            )
        
        # Churn risk recommendations
        if breakdown["churn_risk"]["component_score"] < 50:
            recommendations.append(
                "Address churn risk: Monitor customer frustration signals and intervene proactively."
            )
        
        if not recommendations:
            recommendations.append("Continue current approach - all metrics are within acceptable ranges.")
        
        return recommendations


def calculate_comprehensive_cx_score(
    call_id: str,
    stress_score: Optional[float] = None,
    empathy_alignment_score: Optional[float] = None,
    silence_metrics: Optional[Dict[str, float]] = None,
    speaker_dynamics_path: Optional[str | Path] = None,
    churn_risk_score: Optional[float] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, any]:
    """
    Convenience function to calculate comprehensive CX score.
    
    Args:
        call_id: Call identifier.
        stress_score: Unified stress score (0-1).
        empathy_alignment_score: Empathy alignment score (0-100).
        silence_metrics: Dictionary with silence metrics.
        speaker_dynamics_path: Path to speaker dynamics CSV file.
        churn_risk_score: Early churn risk score (0-100).
        output_path: Optional path to save JSON results.
    
    Returns:
        Dictionary containing comprehensive CX score analysis.
    
    Example:
        >>> result = calculate_comprehensive_cx_score(
        ...     call_id="call_abc123",
        ...     stress_score=0.3,
        ...     empathy_alignment_score=75.0,
        ...     silence_metrics={"silence_ratio": 0.15},
        ...     speaker_dynamics_path="data/processed/speaker_dynamics.csv",
        ...     churn_risk_score=25.0
        ... )
    """
    calculator = ComprehensiveCXScore()
    
    result = calculator.calculate_cx_score(
        call_id=call_id,
        stress_score=stress_score,
        empathy_alignment_score=empathy_alignment_score,
        silence_metrics=silence_metrics,
        speaker_dynamics_path=speaker_dynamics_path,
        churn_risk_score=churn_risk_score,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved comprehensive CX score to: {output_path}")
    
    return result
