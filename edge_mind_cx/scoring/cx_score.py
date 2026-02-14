"""
CX (Customer Experience) score calculation module for EdgeMindCX project.

This module combines stress score, empathy alignment, and silence metrics
to produce a comprehensive CX score (0-100) for call center interactions.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CXScoreCalculator:
    """
    Calculates comprehensive CX score from multiple behavioral metrics.
    
    Combines:
    - Stress score (inverted: lower stress = higher CX)
    - Empathy alignment (higher empathy = higher CX)
    - Silence metrics (optimal silence range = higher CX)
    """
    
    # Optimal silence ratio range (too much or too little is bad)
    OPTIMAL_SILENCE_MIN = 0.1  # 10% silence is acceptable
    OPTIMAL_SILENCE_MAX = 0.3  # 30% silence is acceptable
    
    def __init__(
        self,
        stress_weight: float = 0.4,
        empathy_weight: float = 0.4,
        silence_weight: float = 0.2,
        normalize_weights: bool = True,
    ) -> None:
        """
        Initialize CX score calculator.
        
        Args:
            stress_weight: Weight for stress component (inverted).
                          Default is 0.4.
            empathy_weight: Weight for empathy component.
                           Default is 0.4.
            silence_weight: Weight for silence metrics component.
                           Default is 0.2.
            normalize_weights: If True, automatically normalize weights to sum to 1.0.
                              Default is True.
        """
        self.stress_weight = stress_weight
        self.empathy_weight = empathy_weight
        self.silence_weight = silence_weight
        
        # Validate and normalize weights
        if normalize_weights:
            total = stress_weight + empathy_weight + silence_weight
            if total > 0:
                self.stress_weight = stress_weight / total
                self.empathy_weight = empathy_weight / total
                self.silence_weight = silence_weight / total
                logger.info(
                    f"Weights normalized. Final weights: "
                    f"stress={self.stress_weight:.2f}, "
                    f"empathy={self.empathy_weight:.2f}, "
                    f"silence={self.silence_weight:.2f}"
                )
            else:
                raise ValueError("Sum of weights must be greater than 0")
        else:
            total = stress_weight + empathy_weight + silence_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0 (current sum: {total}). "
                    f"Set normalize_weights=True to auto-normalize."
                )
            if any(w < 0 for w in [stress_weight, empathy_weight, silence_weight]):
                raise ValueError("Weights cannot be negative")
        
        logger.info(
            f"CXScoreCalculator initialized with weights: "
            f"stress={self.stress_weight:.2f}, "
            f"empathy={self.empathy_weight:.2f}, "
            f"silence={self.silence_weight:.2f}"
        )
    
    def calculate_cx_score(
        self,
        stress_score: Optional[float] = None,
        empathy_score: Optional[float] = None,
        silence_ratio: Optional[float] = None,
        silence_duration: Optional[float] = None,
        total_duration: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Calculate comprehensive CX score from input metrics.
        
        Args:
            stress_score: Unified stress score (0-1). Higher = more stress.
                        Default is None (will use neutral 0.5).
            empathy_score: Empathy alignment score (0-1). Higher = better empathy.
                          Default is None (will use neutral 0.5).
            silence_ratio: Ratio of silence to total duration (0-1).
                          Default is None (will calculate from silence_duration/total_duration).
            silence_duration: Total silence duration in seconds.
                             Used if silence_ratio is not provided.
            total_duration: Total call duration in seconds.
                          Required if calculating silence_ratio from silence_duration.
        
        Returns:
            Dictionary containing:
            - cx_score: Final CX score (0-100)
            - stress_component: Stress component score (0-100)
            - empathy_component: Empathy component score (0-100)
            - silence_component: Silence component score (0-100)
            - breakdown: Detailed breakdown of all components
        """
        # Calculate individual components
        stress_component = self._calculate_stress_component(stress_score)
        empathy_component = self._calculate_empathy_component(empathy_score)
        silence_component = self._calculate_silence_component(
            silence_ratio, silence_duration, total_duration
        )
        
        # Weighted combination (0-1 scale)
        cx_score_normalized = (
            self.stress_weight * stress_component
            + self.empathy_weight * empathy_component
            + self.silence_weight * silence_component
        )
        
        # Convert to 0-100 scale
        cx_score = float(np.clip(cx_score_normalized * 100.0, 0.0, 100.0))
        
        # Convert components to 0-100 scale for consistency
        stress_component_100 = stress_component * 100.0
        empathy_component_100 = empathy_component * 100.0
        silence_component_100 = silence_component * 100.0
        
        return {
            "cx_score": cx_score,
            "stress_component": stress_component_100,
            "empathy_component": empathy_component_100,
            "silence_component": silence_component_100,
            "breakdown": {
                "stress": {
                    "raw_score": stress_score if stress_score is not None else 0.5,
                    "component_score": stress_component_100,
                    "weight": self.stress_weight,
                    "contribution": self.stress_weight * stress_component * 100.0,
                    "description": "Lower stress indicates better CX",
                },
                "empathy": {
                    "raw_score": empathy_score if empathy_score is not None else 0.5,
                    "component_score": empathy_component_100,
                    "weight": self.empathy_weight,
                    "contribution": self.empathy_weight * empathy_component * 100.0,
                    "description": "Higher empathy alignment indicates better CX",
                },
                "silence": {
                    "silence_ratio": silence_ratio if silence_ratio is not None else 0.0,
                    "silence_duration": silence_duration if silence_duration is not None else 0.0,
                    "component_score": silence_component_100,
                    "weight": self.silence_weight,
                    "contribution": self.silence_weight * silence_component * 100.0,
                    "description": "Optimal silence range (10-30%) indicates better CX",
                },
            },
        }
    
    def _calculate_stress_component(
        self,
        stress_score: Optional[float],
    ) -> float:
        """
        Calculate stress component for CX score.
        
        Stress is inverted: lower stress = higher CX.
        High stress negatively impacts customer experience.
        
        Args:
            stress_score: Stress score (0-1). Higher = more stress.
        
        Returns:
            Stress component score (0-1), where 1.0 = no stress (best CX).
        """
        if stress_score is None:
            logger.warning("No stress score provided, using neutral 0.5")
            return 0.5
        
        # Invert: lower stress = higher CX
        # stress_score 0.0 (no stress) -> component 1.0 (best)
        # stress_score 1.0 (high stress) -> component 0.0 (worst)
        stress_component = 1.0 - np.clip(stress_score, 0.0, 1.0)
        
        return float(stress_component)
    
    def _calculate_empathy_component(
        self,
        empathy_score: Optional[float],
    ) -> float:
        """
        Calculate empathy component for CX score.
        
        Higher empathy alignment directly improves customer experience.
        
        Args:
            empathy_score: Empathy alignment score (0-1). Higher = better empathy.
        
        Returns:
            Empathy component score (0-1), where 1.0 = perfect empathy (best CX).
        """
        if empathy_score is None:
            logger.warning("No empathy score provided, using neutral 0.5")
            return 0.5
        
        # Direct mapping: higher empathy = higher CX
        empathy_component = np.clip(empathy_score, 0.0, 1.0)
        
        return float(empathy_component)
    
    def _calculate_silence_component(
        self,
        silence_ratio: Optional[float],
        silence_duration: Optional[float],
        total_duration: Optional[float],
    ) -> float:
        """
        Calculate silence component for CX score.
        
        Optimal silence range: 10-30% of call duration.
        Too much silence (>30%) indicates disengagement or awkwardness.
        Too little silence (<10%) indicates rushed conversation.
        
        Args:
            silence_ratio: Ratio of silence to total duration (0-1).
            silence_duration: Total silence duration in seconds.
            total_duration: Total call duration in seconds.
        
        Returns:
            Silence component score (0-1), where 1.0 = optimal silence (best CX).
        """
        # Calculate silence_ratio if not provided
        if silence_ratio is None:
            if silence_duration is not None and total_duration is not None:
                if total_duration > 0:
                    silence_ratio = silence_duration / total_duration
                else:
                    logger.warning("Total duration is 0, using default silence ratio 0.0")
                    silence_ratio = 0.0
            else:
                logger.warning("No silence metrics provided, using neutral score 0.5")
                return 0.5
        
        silence_ratio = np.clip(silence_ratio, 0.0, 1.0)
        
        # Optimal range: 10-30%
        if self.OPTIMAL_SILENCE_MIN <= silence_ratio <= self.OPTIMAL_SILENCE_MAX:
            # Perfect range: score = 1.0
            score = 1.0
        elif silence_ratio < self.OPTIMAL_SILENCE_MIN:
            # Too little silence: linear penalty
            # 0% silence -> score 0.5, 10% silence -> score 1.0
            score = 0.5 + (silence_ratio / self.OPTIMAL_SILENCE_MIN) * 0.5
        else:
            # Too much silence: exponential penalty
            # 30% silence -> score 1.0, 100% silence -> score 0.0
            excess = silence_ratio - self.OPTIMAL_SILENCE_MAX
            max_excess = 1.0 - self.OPTIMAL_SILENCE_MAX
            score = max(0.0, 1.0 - (excess / max_excess))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def update_weights(
        self,
        stress_weight: Optional[float] = None,
        empathy_weight: Optional[float] = None,
        silence_weight: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        """
        Update CX score weights dynamically.
        
        Allows runtime adjustment of weights for different use cases or
        calibration based on validation data.
        
        Args:
            stress_weight: New weight for stress component (if provided).
            empathy_weight: New weight for empathy component (if provided).
            silence_weight: New weight for silence component (if provided).
            normalize: If True, normalize weights to sum to 1.0.
        
        Example:
            >>> calculator = CXScoreCalculator()
            >>> calculator.update_weights(stress_weight=0.5, empathy_weight=0.5)
        """
        if stress_weight is not None:
            self.stress_weight = stress_weight
        if empathy_weight is not None:
            self.empathy_weight = empathy_weight
        if silence_weight is not None:
            self.silence_weight = silence_weight
        
        if normalize:
            total = self.stress_weight + self.empathy_weight + self.silence_weight
            if total > 0:
                self.stress_weight = self.stress_weight / total
                self.empathy_weight = self.empathy_weight / total
                self.silence_weight = self.silence_weight / total
        
        logger.info(
            f"Weights updated: stress={self.stress_weight:.2f}, "
            f"empathy={self.empathy_weight:.2f}, "
            f"silence={self.silence_weight:.2f}"
        )


def calculate_cx_score(
    stress_score: Optional[float] = None,
    empathy_score: Optional[float] = None,
    silence_ratio: Optional[float] = None,
    silence_duration: Optional[float] = None,
    total_duration: Optional[float] = None,
    stress_weight: float = 0.4,
    empathy_weight: float = 0.4,
    silence_weight: float = 0.2,
) -> Dict[str, any]:
    """
    Wrapper function to quickly calculate CX score.
    
    Convenience function for simple CX score calculation.
    
    Args:
        stress_score: Unified stress score (0-1).
        empathy_score: Empathy alignment score (0-1).
        silence_ratio: Ratio of silence to total duration (0-1).
        silence_duration: Total silence duration in seconds.
        total_duration: Total call duration in seconds.
        stress_weight: Weight for stress component. Default is 0.4.
        empathy_weight: Weight for empathy component. Default is 0.4.
        silence_weight: Weight for silence component. Default is 0.2.
    
    Returns:
        Dictionary containing cx_score and breakdown.
    
    Example:
        >>> result = calculate_cx_score(
        ...     stress_score=0.3,
        ...     empathy_score=0.8,
        ...     silence_ratio=0.15
        ... )
        >>> print(f"CX Score: {result['cx_score']:.1f}/100")
    """
    calculator = CXScoreCalculator(
        stress_weight=stress_weight,
        empathy_weight=empathy_weight,
        silence_weight=silence_weight,
    )
    
    return calculator.calculate_cx_score(
        stress_score=stress_score,
        empathy_score=empathy_score,
        silence_ratio=silence_ratio,
        silence_duration=silence_duration,
        total_duration=total_duration,
    )
