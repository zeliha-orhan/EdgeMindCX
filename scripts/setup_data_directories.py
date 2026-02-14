"""
Setup data directories for EdgeMindCX project.

Creates required data directories if they don't exist.
"""

import os
from pathlib import Path

# Define required directories
REQUIRED_DIRECTORIES = [
    "data/raw/audio/call_center",
    "data/processed/audio",
    "data/processed/transcripts",
    "data/processed/diarization",
    "data/processed/behavior",
    "data/processed/reports",
]


def setup_data_directories() -> None:
    """
    Create required data directories if they don't exist.
    
    Prints validation output showing which directories were created
    and which already existed.
    """
    created_dirs = []
    existing_dirs = []
    
    for dir_path in REQUIRED_DIRECTORIES:
        path = Path(dir_path)
        
        if path.exists():
            existing_dirs.append(dir_path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
    
    # Print validation output
    print("=" * 60)
    print("EDGEMINDCX - DATA DIRECTORY SETUP")
    print("=" * 60)
    print()
    
    if created_dirs:
        print("[CREATED] Directories:")
        for dir_path in created_dirs:
            print(f"  - {dir_path}")
        print()
    else:
        print("[INFO] No new directories created (all already exist)")
        print()
    
    if existing_dirs:
        print("[EXISTING] Directories:")
        for dir_path in existing_dirs:
            print(f"  - {dir_path}")
        print()
    
    print("=" * 60)
    print(f"Total directories checked: {len(REQUIRED_DIRECTORIES)}")
    print(f"Created: {len(created_dirs)}")
    print(f"Already existed: {len(existing_dirs)}")
    print("=" * 60)


if __name__ == "__main__":
    setup_data_directories()
