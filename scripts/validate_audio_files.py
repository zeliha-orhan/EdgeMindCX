"""
Audio file validation script for EdgeMindCX project.

Scans .wav files in data/raw/audio/call_center/ and validates:
- File names
- Duration (seconds)
- Sample rate
- Channel information
"""

import wave
from pathlib import Path


def validate_audio_files(audio_dir: str | Path = "data/raw/audio/call_center") -> None:
    """
    Validate audio files in the specified directory.
    
    Args:
        audio_dir: Directory containing .wav audio files.
    """
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        print(f"ERROR: Directory not found: {audio_dir}")
        return
    
    # Find all .wav files
    wav_files = list(audio_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in: {audio_dir}")
        return
    
    print("=" * 80)
    print("EDGEMINDCX - AUDIO FILE VALIDATION")
    print("=" * 80)
    print(f"Directory: {audio_dir}")
    print(f"Total files found: {len(wav_files)}")
    print()
    
    # Validate each file
    for i, audio_file in enumerate(sorted(wav_files), 1):
        print(f"[{i}/{len(wav_files)}] {audio_file.name}")
        print("-" * 80)
        
        try:
            # Get file info using wave module
            with wave.open(str(audio_file), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                duration = frames / float(sample_rate)
            
            # Format duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            milliseconds = int((duration % 1) * 1000)
            
            print(f"  Duration: {duration:.2f}s ({minutes}m {seconds}s {milliseconds}ms)")
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Channels: {channels} ({'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels}-channel'})")
            print(f"  Frames: {frames:,}")
            print(f"  Sample Width: {sample_width} bytes")
            
            # Validate against expected values
            issues = []
            if sample_rate != 16000:
                issues.append(f"Sample rate is {sample_rate} Hz (expected 16000 Hz)")
            
            if channels != 1:
                issues.append(f"Channels: {channels} (expected 1 - mono)")
            
            if issues:
                print(f"  [WARNING] Issues found:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"  [OK] File format is valid (16kHz, mono)")
            
        except Exception as e:
            print(f"  [ERROR] Could not read file: {e}")
        
        print()
    
    print("=" * 80)
    print("Validation complete")
    print("=" * 80)


if __name__ == "__main__":
    validate_audio_files()
