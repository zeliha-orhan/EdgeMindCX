"""
EdgeMindCX - Customer Experience Analysis for Call Centers

Main package for behavioral analysis and CX scoring from call center audio.
Audio-first architecture: processes .wav files directly without CSV metadata.
"""

from edge_mind_cx.audio_loader import AudioDataLoader
from edge_mind_cx.pipeline import EdgeMindCXPipeline, run_pipeline

__all__ = [
    "EdgeMindCXPipeline",
    "run_pipeline",
    "AudioDataLoader",
]

__version__ = "0.2.0"
