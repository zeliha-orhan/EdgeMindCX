"""
Transcription module for EdgeMindCX.

Provides local Whisper-based speech-to-text transcription and speaker diarization.
"""

from edge_mind_cx.transcription.diarization import (
    SpeakerDiarization,
    diarize_call,
)
from edge_mind_cx.transcription.pipeline import (
    WhisperTranscriptionPipeline,
    transcribe_validated_audio,
)
from edge_mind_cx.transcription.transcription import (
    TranscriptionResult,
    WhisperTranscriber,
)

__all__ = [
    "WhisperTranscriber",
    "TranscriptionResult",
    "WhisperTranscriptionPipeline",
    "transcribe_validated_audio",
    "SpeakerDiarization",
    "diarize_call",
]
