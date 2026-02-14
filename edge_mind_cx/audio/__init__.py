"""
Audio processing module for EdgeMindCX.

Provides audio ingestion, validation, and normalization capabilities.
"""

from edge_mind_cx.audio.ingestion import AudioIngestion, ingest_audio_files

__all__ = [
    "AudioIngestion",
    "ingest_audio_files",
]
