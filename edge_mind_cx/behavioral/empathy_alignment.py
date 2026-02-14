"""
Empathy alignment module for EdgeMindCX project.

This module calculates empathy alignment between customer and agent based on:
- Speech rate alignment
- Emotion alignment
- Response latency (extensible for future use)
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmpathyAlignmentAnalyzer:
    """
    Analyzes empathy alignment between customer and agent.
    
    Empathy alignment measures how well the agent adapts to the customer's
    communication style and emotional state. Higher alignment indicates
    better rapport and understanding.
    """
    
    def __init__(
        self,
        speech_rate_weight: float = 0.4,
        emotion_weight: float = 0.5,
        response_latency_weight: float = 0.1,
        normalize_weights: bool = True,
    ) -> None:
        """
        Initialize empathy alignment analyzer.
        
        Args:
            speech_rate_weight: Weight for speech rate alignment component.
                              Default is 0.4.
            emotion_weight: Weight for emotion alignment component.
                          Default is 0.5.
            response_latency_weight: Weight for response latency component.
                                    Default is 0.1 (currently minimal impact).
            normalize_weights: If True, automatically normalize weights to sum to 1.0.
                              Default is True.
        """
        self.speech_rate_weight = speech_rate_weight
        self.emotion_weight = emotion_weight
        self.response_latency_weight = response_latency_weight
        
        # Validate and normalize weights
        if normalize_weights:
            total = speech_rate_weight + emotion_weight + response_latency_weight
            if total > 0:
                self.speech_rate_weight = speech_rate_weight / total
                self.emotion_weight = emotion_weight / total
                self.response_latency_weight = response_latency_weight / total
                logger.info(
                    f"Weights normalized. Final weights: "
                    f"speech_rate={self.speech_rate_weight:.2f}, "
                    f"emotion={self.emotion_weight:.2f}, "
                    f"latency={self.response_latency_weight:.2f}"
                )
            else:
                raise ValueError("Sum of weights must be greater than 0")
        else:
            total = speech_rate_weight + emotion_weight + response_latency_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0 (current sum: {total}). "
                    f"Set normalize_weights=True to auto-normalize."
                )
            if any(w < 0 for w in [speech_rate_weight, emotion_weight, response_latency_weight]):
                raise ValueError("Weights cannot be negative")
        
        logger.info(
            f"EmpathyAlignmentAnalyzer initialized with weights: "
            f"speech_rate={self.speech_rate_weight:.2f}, "
            f"emotion={self.emotion_weight:.2f}, "
            f"latency={self.response_latency_weight:.2f}"
        )
    
    def analyze_alignment(
        self,
        customer_features: Optional[Dict[str, float]] = None,
        agent_features: Optional[Dict[str, float]] = None,
        customer_emotion_score: Optional[float] = None,
        agent_emotion_score: Optional[float] = None,
        response_latency: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Analyze empathy alignment between customer and agent.
        
        Args:
            customer_features: Dictionary of customer audio features.
                              Expected keys: speech_rate, pitch_mean, etc.
            agent_features: Dictionary of agent audio features.
                           Expected keys: speech_rate, pitch_mean, etc.
            customer_emotion_score: Customer emotion/stress score (0-1).
                                   Higher = more negative/stressed.
            agent_emotion_score: Agent emotion/stress score (0-1).
                                Higher = more negative/stressed.
            response_latency: Response latency in seconds (optional, for future use).
                            Lower latency = better alignment.
        
        Returns:
            Dictionary containing:
            - empathy_score: Overall empathy alignment score (0-1)
            - speech_rate_alignment: Speech rate alignment score (0-1)
            - emotion_alignment: Emotion alignment score (0-1)
            - response_latency_score: Response latency score (0-1)
            - explanation: Short explanation text
        """
        # Calculate individual alignment components
        speech_rate_align = self._calculate_speech_rate_alignment(
            customer_features, agent_features
        )
        emotion_align = self._calculate_emotion_alignment(
            customer_emotion_score, agent_emotion_score
        )
        latency_score = self._calculate_latency_score(response_latency)
        
        # Weighted combination
        empathy_score = (
            self.speech_rate_weight * speech_rate_align
            + self.emotion_weight * emotion_align
            + self.response_latency_weight * latency_score
        )
        
        # Clamp to [0, 1]
        empathy_score = float(np.clip(empathy_score, 0.0, 1.0))
        
        # Generate explanation
        explanation = self._generate_explanation(
            empathy_score,
            speech_rate_align,
            emotion_align,
            latency_score,
        )
        
        return {
            "empathy_score": empathy_score,
            "speech_rate_alignment": speech_rate_align,
            "emotion_alignment": emotion_align,
            "response_latency_score": latency_score,
            "explanation": explanation,
            "component_scores": {
                "speech_rate": {
                    "score": speech_rate_align,
                    "weight": self.speech_rate_weight,
                    "contribution": self.speech_rate_weight * speech_rate_align,
                },
                "emotion": {
                    "score": emotion_align,
                    "weight": self.emotion_weight,
                    "contribution": self.emotion_weight * emotion_align,
                },
                "latency": {
                    "score": latency_score,
                    "weight": self.response_latency_weight,
                    "contribution": self.response_latency_weight * latency_score,
                },
            },
        }
    
    def _calculate_speech_rate_alignment(
        self,
        customer_features: Optional[Dict[str, float]],
        agent_features: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate speech rate alignment between customer and agent.
        
        Alignment is measured as similarity in speech rate. Agents who
        match customer's pace show better empathy and understanding.
        
        Args:
            customer_features: Customer audio features.
            agent_features: Agent audio features.
        
        Returns:
            Speech rate alignment score (0-1), where 1.0 = perfect alignment.
        """
        if customer_features is None or agent_features is None:
            logger.warning(
                "Missing audio features for speech rate alignment, using default 0.5"
            )
            return 0.5
        
        customer_rate = customer_features.get("speech_rate", 0.5)
        agent_rate = agent_features.get("speech_rate", 0.5)
        
        # Calculate similarity using inverse distance
        # Closer rates = higher alignment
        rate_diff = abs(customer_rate - agent_rate)
        
        # Normalize: perfect alignment (diff=0) = 1.0, max diff (1.0) = 0.0
        # Using exponential decay for smoother transition
        alignment = np.exp(-5.0 * rate_diff)
        
        return float(np.clip(alignment, 0.0, 1.0))
    
    def _calculate_emotion_alignment(
        self,
        customer_emotion_score: Optional[float],
        agent_emotion_score: Optional[float],
    ) -> float:
        """
        Calculate emotion alignment between customer and agent.
        
        Emotion alignment measures how well the agent's emotional response
        matches the customer's emotional state. For empathy, agents should
        acknowledge customer's emotions but not mirror negative emotions
        (which would amplify stress). Instead, they should show understanding
        while maintaining a calmer, supportive tone.
        
        Optimal alignment: Agent is calmer than customer but not too distant.
        
        Args:
            customer_emotion_score: Customer emotion/stress score (0-1).
            agent_emotion_score: Agent emotion/stress score (0-1).
        
        Returns:
            Emotion alignment score (0-1), where 1.0 = optimal alignment.
        """
        if customer_emotion_score is None or agent_emotion_score is None:
            logger.warning(
                "Missing emotion scores for emotion alignment, using default 0.5"
            )
            return 0.5
        
        # Normalize scores
        customer_emotion = np.clip(customer_emotion_score, 0.0, 1.0)
        agent_emotion = np.clip(agent_emotion_score, 0.0, 1.0)
        
        # Optimal alignment: Agent is 0.2-0.4 points calmer than customer
        # This shows understanding without amplifying negativity
        emotion_diff = customer_emotion - agent_emotion
        
        # Ideal difference range: 0.2 to 0.4 (agent calmer)
        ideal_min = 0.2
        ideal_max = 0.4
        
        if ideal_min <= emotion_diff <= ideal_max:
            # Perfect alignment zone
            alignment = 1.0
        elif emotion_diff < ideal_min:
            # Agent too similar or more stressed (not empathetic)
            # Penalize heavily if agent is more stressed
            if emotion_diff < 0:
                alignment = max(0.0, 1.0 - 3.0 * abs(emotion_diff))
            else:
                alignment = max(0.0, 1.0 - 2.0 * (ideal_min - emotion_diff))
        else:
            # Agent too calm (distant, not empathetic)
            alignment = max(0.0, 1.0 - 2.0 * (emotion_diff - ideal_max))
        
        return float(np.clip(alignment, 0.0, 1.0))
    
    def _calculate_latency_score(
        self,
        response_latency: Optional[float],
    ) -> float:
        """
        Calculate response latency score.
        
        Lower latency (faster responses) indicates better engagement
        and empathy. This is a placeholder for future expansion.
        
        Args:
            response_latency: Response latency in seconds.
                           None if not available.
        
        Returns:
            Latency score (0-1), where 1.0 = optimal (low latency).
        """
        if response_latency is None:
            # Default to neutral score if not available
            return 0.7
        
        # Optimal latency: < 1 second
        # Acceptable: 1-3 seconds
        # Poor: > 3 seconds
        
        if response_latency <= 1.0:
            score = 1.0
        elif response_latency <= 3.0:
            # Linear decay from 1.0 to 0.5
            score = 1.0 - 0.25 * (response_latency - 1.0)
        else:
            # Exponential decay for longer latencies
            score = 0.5 * np.exp(-0.5 * (response_latency - 3.0))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _generate_explanation(
        self,
        empathy_score: float,
        speech_rate_align: float,
        emotion_align: float,
        latency_score: float,
    ) -> str:
        """
        Generate short explanation text for empathy score.
        
        Args:
            empathy_score: Overall empathy score (0-1).
            speech_rate_align: Speech rate alignment score (0-1).
            emotion_align: Emotion alignment score (0-1).
            latency_score: Response latency score (0-1).
        
        Returns:
            Short explanation text.
        """
        # Determine overall level
        if empathy_score >= 0.8:
            level = "excellent"
            base_text = "Excellent empathy alignment"
        elif empathy_score >= 0.6:
            level = "good"
            base_text = "Good empathy alignment"
        elif empathy_score >= 0.4:
            level = "moderate"
            base_text = "Moderate empathy alignment"
        else:
            level = "low"
            base_text = "Low empathy alignment"
        
        # Identify strongest and weakest components
        components = {
            "speech rate": speech_rate_align,
            "emotion": emotion_align,
            "response time": latency_score,
        }
        
        strongest = max(components.items(), key=lambda x: x[1])
        weakest = min(components.items(), key=lambda x: x[1])
        
        # Build explanation
        explanation = f"{base_text}. "
        
        if strongest[1] >= 0.7:
            explanation += f"Strong {strongest[0]} matching. "
        
        if weakest[1] < 0.5:
            explanation += f"Opportunity to improve {weakest[0]} alignment."
        else:
            explanation += "All components show reasonable alignment."
        
        return explanation.strip()
    
    def update_weights(
        self,
        speech_rate_weight: Optional[float] = None,
        emotion_weight: Optional[float] = None,
        response_latency_weight: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        """
        Update alignment weights dynamically.
        
        Allows runtime adjustment of weights for different use cases or
        calibration based on validation data.
        
        Args:
            speech_rate_weight: New weight for speech rate alignment (if provided).
            emotion_weight: New weight for emotion alignment (if provided).
            response_latency_weight: New weight for response latency (if provided).
            normalize: If True, normalize weights to sum to 1.0.
        
        Example:
            >>> analyzer = EmpathyAlignmentAnalyzer()
            >>> analyzer.update_weights(emotion_weight=0.7, speech_rate_weight=0.3)
        """
        if speech_rate_weight is not None:
            self.speech_rate_weight = speech_rate_weight
        if emotion_weight is not None:
            self.emotion_weight = emotion_weight
        if response_latency_weight is not None:
            self.response_latency_weight = response_latency_weight
        
        if normalize:
            total = (
                self.speech_rate_weight
                + self.emotion_weight
                + self.response_latency_weight
            )
            if total > 0:
                self.speech_rate_weight = self.speech_rate_weight / total
                self.emotion_weight = self.emotion_weight / total
                self.response_latency_weight = self.response_latency_weight / total
        
        logger.info(
            f"Weights updated: speech_rate={self.speech_rate_weight:.2f}, "
            f"emotion={self.emotion_weight:.2f}, "
            f"latency={self.response_latency_weight:.2f}"
        )


def calculate_empathy_alignment(
    customer_features: Optional[Dict[str, float]] = None,
    agent_features: Optional[Dict[str, float]] = None,
    customer_emotion_score: Optional[float] = None,
    agent_emotion_score: Optional[float] = None,
    response_latency: Optional[float] = None,
) -> Dict[str, any]:
    """
    Wrapper function to quickly calculate empathy alignment.
    
    Convenience function for simple empathy alignment calculation.
    
    Args:
        customer_features: Dictionary of customer audio features.
        agent_features: Dictionary of agent audio features.
        customer_emotion_score: Customer emotion/stress score (0-1).
        agent_emotion_score: Agent emotion/stress score (0-1).
        response_latency: Response latency in seconds (optional).
    
    Returns:
        Dictionary containing empathy_score and explanation.
    
    Example:
        >>> result = calculate_empathy_alignment(
        ...     customer_features=customer_audio_feats,
        ...     agent_features=agent_audio_feats,
        ...     customer_emotion_score=0.8,
        ...     agent_emotion_score=0.4
        ... )
        >>> print(f"Empathy: {result['empathy_score']:.2f}")
        >>> print(result['explanation'])
    """
    analyzer = EmpathyAlignmentAnalyzer()
    
    return analyzer.analyze_alignment(
        customer_features=customer_features,
        agent_features=agent_features,
        customer_emotion_score=customer_emotion_score,
        agent_emotion_score=agent_emotion_score,
        response_latency=response_latency,
    )
