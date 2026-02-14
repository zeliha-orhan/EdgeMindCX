"""
Text-based stress analysis module for EdgeMindCX project.

This module analyzes transcribed text to detect stress signals using
HuggingFace sentiment and emotion models. Measures negativity and intensity
to produce a stress score.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)

# Global cache for stress analysis models (singleton pattern)
_stress_models_cache: Dict[str, any] = {}


def load_stress_models(
    sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    device: Optional[Union[int, str]] = None,
) -> Dict[str, any]:
    """
    Load HuggingFace models for sentiment and emotion analysis.
    
    This function loads pre-trained models separately to allow for
    flexible model selection and resource management.
    
    Args:
        sentiment_model_name: HuggingFace model identifier for sentiment analysis.
                            Default: cardiffnlp/twitter-roberta-base-sentiment-latest
                            (good for social/casual text, works well for call center transcripts)
        emotion_model_name: HuggingFace model identifier for emotion analysis.
                           Default: j-hartmann/emotion-english-distilroberta-base
                           (detects joy, sadness, anger, fear, surprise, disgust, neutral)
        device: Device to run models on ('cpu', 'cuda', or device index).
                If None, auto-detects (prefers GPU if available).
    
    Returns:
        Dictionary containing:
        - sentiment_pipeline: HuggingFace pipeline for sentiment analysis
        - emotion_pipeline: HuggingFace pipeline for emotion analysis
        - sentiment_tokenizer: Tokenizer for sentiment model
        - emotion_tokenizer: Tokenizer for emotion model
    
    Example:
        >>> models = load_stress_models()
        >>> sentiment_result = models["sentiment_pipeline"]("I'm very frustrated with this service")
    """
    # Create cache key from model names and device
    cache_key = f"{sentiment_model_name}_{emotion_model_name}_{device or 'auto'}"
    
    if cache_key in _stress_models_cache:
        logger.info(f"Using cached stress analysis models")
        return _stress_models_cache[cache_key]
    
    logger.info(f"Loading sentiment model: {sentiment_model_name} (first time, will be cached)")
    
    try:
        # Load sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model_name,
            device=device,
            return_all_scores=True,  # Get scores for all labels
        )
        
        logger.info(f"Sentiment model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading sentiment model: {e}")
        raise
    
    logger.info(f"Loading emotion model: {emotion_model_name} (first time, will be cached)")
    
    try:
        # Load emotion analysis pipeline
        emotion_pipeline = pipeline(
            "text-classification",
            model=emotion_model_name,
            device=device,
            return_all_scores=True,  # Get scores for all emotions
        )
        
        logger.info(f"Emotion model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading emotion model: {emotion_model_name}")
        raise
    
    # Also load tokenizers for potential direct model access
    try:
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    except Exception as e:
        logger.warning(f"Could not load tokenizers: {e}")
        sentiment_tokenizer = None
        emotion_tokenizer = None
    
    models_dict = {
        "sentiment_pipeline": sentiment_pipeline,
        "emotion_pipeline": emotion_pipeline,
        "sentiment_tokenizer": sentiment_tokenizer,
        "emotion_tokenizer": emotion_tokenizer,
    }
    
    # Cache the models
    _stress_models_cache[cache_key] = models_dict
    logger.info(f"Stress analysis models loaded and cached successfully")
    
    return models_dict


class TextStressAnalyzer:
    """
    Analyzes text transcripts to detect stress signals.
    
    Uses HuggingFace sentiment and emotion models to measure:
    - Negativity: Negative sentiment and negative emotions
    - Intensity: Strength of emotional expression
    
    Combines these measures into a stress score (0-1).
    """
    
    # Negative emotion labels (stress indicators)
    NEGATIVE_EMOTIONS = ["sadness", "anger", "fear", "disgust"]
    
    # Positive emotion labels (stress reducers)
    POSITIVE_EMOTIONS = ["joy", "surprise"]
    
    def __init__(
        self,
        models: Optional[Dict[str, any]] = None,
        sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[Union[int, str]] = None,
        negativity_weight: float = 0.6,
        intensity_weight: float = 0.4,
    ) -> None:
        """
        Initialize text stress analyzer.
        
        Args:
            models: Pre-loaded models dictionary from load_stress_models().
                   If None, models will be loaded automatically.
            sentiment_model_name: HuggingFace model for sentiment analysis.
            emotion_model_name: HuggingFace model for emotion analysis.
            device: Device to run models on.
            negativity_weight: Weight for negativity component in stress score (0-1).
                              Default is 0.6.
            intensity_weight: Weight for intensity component in stress score (0-1).
                             Default is 0.4.
                             Note: weights should sum to 1.0
        """
        if models is None:
            logger.info("Loading models automatically...")
            models = load_stress_models(
                sentiment_model_name=sentiment_model_name,
                emotion_model_name=emotion_model_name,
                device=device,
            )
        
        self.sentiment_pipeline = models["sentiment_pipeline"]
        self.emotion_pipeline = models["emotion_pipeline"]
        
        # Validate weights
        if abs(negativity_weight + intensity_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights don't sum to 1.0 (sum={negativity_weight + intensity_weight}). "
                f"Normalizing..."
            )
            total = negativity_weight + intensity_weight
            negativity_weight = negativity_weight / total
            intensity_weight = intensity_weight / total
        
        self.negativity_weight = negativity_weight
        self.intensity_weight = intensity_weight
        
        logger.info(
            f"TextStressAnalyzer initialized with weights: "
            f"negativity={negativity_weight}, intensity={intensity_weight}"
        )
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text and return stress score and components.
        
        Args:
            text: Input text to analyze (transcription).
        
        Returns:
            Dictionary containing:
            - text_stress_score: Overall stress score (0-1)
            - negativity_score: Negativity component (0-1)
            - intensity_score: Intensity component (0-1)
            - sentiment_label: Predicted sentiment label
            - emotion_label: Predicted emotion label
            - negative_emotion_score: Combined negative emotion score
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for stress analysis")
            return {
                "text_stress_score": 0.0,
                "negativity_score": 0.0,
                "intensity_score": 0.0,
                "sentiment_label": "neutral",
                "emotion_label": "neutral",
                "negative_emotion_score": 0.0,
            }
        
        logger.debug(f"Analyzing text for stress: {text[:50]}...")
        
        # Get sentiment analysis
        sentiment_result = self._analyze_sentiment(text)
        
        # Get emotion analysis
        emotion_result = self._analyze_emotion(text)
        
        # Calculate negativity score
        negativity_score = self._calculate_negativity(
            sentiment_result, emotion_result
        )
        
        # Calculate intensity score
        intensity_score = self._calculate_intensity(
            sentiment_result, emotion_result
        )
        
        # Combine into stress score
        stress_score = (
            self.negativity_weight * negativity_score
            + self.intensity_weight * intensity_score
        )
        
        # Clamp to [0, 1]
        stress_score = max(0.0, min(1.0, stress_score))
        
        return {
            "text_stress_score": float(stress_score),
            "negativity_score": float(negativity_score),
            "intensity_score": float(intensity_score),
            "sentiment_label": sentiment_result["label"],
            "emotion_label": emotion_result["label"],
            "negative_emotion_score": float(emotion_result["negative_score"]),
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text.
        
        Returns:
            Dictionary with sentiment analysis results.
        """
        try:
            results = self.sentiment_pipeline(text)
            
            # Find highest scoring label
            # Results format: [{"label": "LABEL", "score": 0.xx}, ...]
            best_result = max(results[0], key=lambda x: x["score"])
            
            # Extract all scores for negativity calculation
            scores = {item["label"].lower(): item["score"] for item in results[0]}
            
            return {
                "label": best_result["label"],
                "score": best_result["score"],
                "scores": scores,
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "label": "neutral",
                "score": 0.5,
                "scores": {"positive": 0.5, "negative": 0.5, "neutral": 0.5},
            }
    
    def _analyze_emotion(self, text: str) -> Dict[str, any]:
        """
        Analyze emotions in text.
        
        Args:
            text: Input text.
        
        Returns:
            Dictionary with emotion analysis results.
        """
        try:
            results = self.emotion_pipeline(text)
            
            # Find highest scoring emotion
            best_result = max(results[0], key=lambda x: x["score"])
            
            # Extract all emotion scores
            emotion_scores = {
                item["label"].lower(): item["score"] for item in results[0]
            }
            
            # Calculate negative emotion score (sum of negative emotions)
            negative_score = sum(
                emotion_scores.get(emotion, 0.0)
                for emotion in self.NEGATIVE_EMOTIONS
            )
            
            # Calculate positive emotion score
            positive_score = sum(
                emotion_scores.get(emotion, 0.0)
                for emotion in self.POSITIVE_EMOTIONS
            )
            
            return {
                "label": best_result["label"],
                "score": best_result["score"],
                "scores": emotion_scores,
                "negative_score": negative_score,
                "positive_score": positive_score,
            }
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {
                "label": "neutral",
                "score": 0.5,
                "scores": {},
                "negative_score": 0.0,
                "positive_score": 0.0,
            }
    
    def _calculate_negativity(
        self,
        sentiment_result: Dict[str, any],
        emotion_result: Dict[str, any],
    ) -> float:
        """
        Calculate negativity score from sentiment and emotion results.
        
        Negativity combines:
        - Negative sentiment probability
        - Negative emotion probabilities (sadness, anger, fear, disgust)
        
        Args:
            sentiment_result: Sentiment analysis results.
            emotion_result: Emotion analysis results.
        
        Returns:
            Negativity score (0-1), where 1.0 is maximum negativity.
        """
        # Get negative sentiment score
        sentiment_scores = sentiment_result.get("scores", {})
        negative_sentiment = sentiment_scores.get("negative", 0.0)
        
        # If model uses LABEL_0, LABEL_1 format, try to infer
        if "negative" not in sentiment_scores:
            # Try common label formats
            for key in sentiment_scores.keys():
                if "neg" in key.lower():
                    negative_sentiment = sentiment_scores[key]
                    break
        
        # Get negative emotion score (already calculated)
        negative_emotion = emotion_result.get("negative_score", 0.0)
        
        # Combine: weighted average (emotion slightly more important for stress)
        negativity = 0.4 * negative_sentiment + 0.6 * negative_emotion
        
        return float(np.clip(negativity, 0.0, 1.0))
    
    def _calculate_intensity(
        self,
        sentiment_result: Dict[str, any],
        emotion_result: Dict[str, any],
    ) -> float:
        """
        Calculate intensity score from sentiment and emotion results.
        
        Intensity measures how strong the emotional expression is,
        regardless of positive/negative. High intensity + negativity = stress.
        
        Args:
            sentiment_result: Sentiment analysis results.
            emotion_result: Emotion analysis results.
        
        Returns:
            Intensity score (0-1), where 1.0 is maximum intensity.
        """
        # Sentiment confidence (how certain the model is)
        sentiment_confidence = sentiment_result.get("score", 0.5)
        
        # Emotion confidence (how certain the model is about emotion)
        emotion_confidence = emotion_result.get("score", 0.5)
        
        # Maximum emotion score (strongest emotion, regardless of type)
        emotion_scores = emotion_result.get("scores", {})
        if emotion_scores:
            max_emotion = max(emotion_scores.values())
        else:
            max_emotion = 0.5
        
        # Combine: average of confidence measures
        intensity = (
            0.3 * sentiment_confidence
            + 0.3 * emotion_confidence
            + 0.4 * max_emotion
        )
        
        return float(np.clip(intensity, 0.0, 1.0))


def analyze_text_stress(
    text: str,
    models: Optional[Dict[str, any]] = None,
    **analyzer_kwargs,
) -> float:
    """
    Wrapper function to quickly get stress score from text.
    
    Convenience function for simple stress score extraction.
    
    Args:
        text: Input text to analyze.
        models: Pre-loaded models. If None, will be loaded automatically.
        **analyzer_kwargs: Additional arguments to pass to TextStressAnalyzer.
    
    Returns:
        Text stress score (0-1).
    
    Example:
        >>> score = analyze_text_stress("I'm extremely frustrated with this service!")
        >>> print(f"Stress score: {score:.2f}")
    """
    analyzer = TextStressAnalyzer(models=models, **analyzer_kwargs)
    result = analyzer.analyze(text)
    return result["text_stress_score"]
