"""
End-to-end pipeline for EdgeMindCX project.

Orchestrates the complete workflow from audio loading to CX score calculation.
Audio-first architecture: processes .wav files directly without CSV metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from edge_mind_cx.analysis.audio_features import AudioFeatureExtractor
from edge_mind_cx.analysis.opensmile_features import extract_egemaps_features
from edge_mind_cx.audio_loader import AudioDataLoader
from edge_mind_cx.behavioral.empathy_alignment import EmpathyAlignmentAnalyzer
from edge_mind_cx.behavioral.text_stress_analyzer import TextStressAnalyzer
from edge_mind_cx.scoring.cx_score import CXScoreCalculator
from edge_mind_cx.scoring.stress_score_fusion import StressScoreFusion
from edge_mind_cx.transcription.transcription import WhisperTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EdgeMindCXPipeline:
    """
    End-to-end pipeline for EdgeMindCX analysis.
    
    Audio-first architecture: processes .wav files directly without CSV metadata.
    
    Orchestrates:
    1. Audio loading from directory
    2. Whisper transcription
    3. Feature extraction (audio + openSMILE)
    4. Score calculation (stress + empathy + CX)
    5. Results output
    """
    
    def __init__(
        self,
        audio_dir: str | Path = "data/raw/audio/call_center",
        whisper_model_size: str = "small",
        output_dir: Optional[str | Path] = None,
        save_results: bool = True,
        print_results: bool = True,
    ) -> None:
        """
        Initialize EdgeMindCX pipeline.
        
        Args:
            audio_dir: Directory containing .wav audio files.
                     Default is "data/raw/audio/call_center".
            whisper_model_size: Whisper model size ('small' or 'medium').
            output_dir: Directory to save results. If None, uses 'data/processed'.
            save_results: Whether to save results to JSON. Default is True.
            print_results: Whether to print results to console. Default is True.
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.save_results = save_results
        self.print_results = print_results
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.data_loader = AudioDataLoader(audio_dir=audio_dir)
        
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size)
        
        self.audio_feature_extractor = AudioFeatureExtractor()
        
        self.stress_analyzer = TextStressAnalyzer()
        
        self.stress_fusion = StressScoreFusion()
        
        self.empathy_analyzer = EmpathyAlignmentAnalyzer()
        
        self.cx_calculator = CXScoreCalculator()
        
        logger.info("Pipeline initialized successfully")
    
    def process_sample(
        self,
        sample: Dict,
    ) -> Dict[str, any]:
        """
        Process a single sample through the complete pipeline.
        
        Args:
            sample: Sample dictionary from data loader containing:
                   call_id, waveform, sample_rate, audio_path, filename
        
        Returns:
            Dictionary containing all analysis results and scores.
        """
        results = {
            "call_id": sample.get("call_id"),
            "filename": sample.get("filename"),
            "audio_path": sample.get("audio_path"),
        }
        
        audio_path = Path(sample.get("audio_path")) if sample.get("audio_path") else None
        
        # Step 1: Audio features (librosa)
        logger.debug("Extracting audio features...")
        audio_features = self.audio_feature_extractor.extract_features(
            waveform=sample["waveform"],
            sample_rate=sample["sample_rate"],
        )
        results["audio_features"] = audio_features
        
        # Step 2: Transcription
        logger.debug("Transcribing audio...")
        transcription_result = self.transcriber.transcribe_from_waveform(
            waveform=sample["waveform"],
            sample_rate=sample["sample_rate"],
        )
        results["transcription"] = {
            "text": transcription_result.text,
            "num_segments": len(transcription_result.segments),
            "segments": [
                {
                    "text": s.get("text", ""),
                    "start_time": s.get("start_time", 0.0),
                    "end_time": s.get("end_time", 0.0),
                }
                for s in transcription_result.segments
            ],
        }
        
        # Step 3: NLP stress analysis
        logger.debug("Analyzing text stress...")
        nlp_stress = self.stress_analyzer.analyze(transcription_result.text)
        results["nlp_stress"] = nlp_stress["text_stress_score"]
        
        # Step 4: openSMILE features
        opensmile_features = None
        if audio_path and audio_path.exists():
            logger.debug("Extracting openSMILE features...")
            try:
                opensmile_features = extract_egemaps_features(
                    audio_path=audio_path,
                    return_dict=True,
                )
                results["opensmile_features"] = opensmile_features
            except Exception as e:
                logger.warning(f"Could not extract openSMILE features: {e}")
        else:
            logger.warning("Audio path not available for openSMILE extraction")
        
        # Step 5: Stress score fusion
        logger.debug("Calculating unified stress score...")
        stress_result = self.stress_fusion.compute_final_score(
            audio_features=audio_features,
            opensmile_features=opensmile_features,
            nlp_stress_score=nlp_stress["text_stress_score"],
        )
        results["stress_score"] = stress_result["stress_score"]
        
        # Step 6: Empathy alignment (requires customer and agent data)
        # For single sample, we'll use default/neutral values
        # In real scenario, you'd have separate customer/agent segments
        logger.debug("Calculating empathy alignment...")
        empathy_result = self.empathy_analyzer.analyze_alignment(
            customer_features=audio_features,
            agent_features=None,  # Would need separate agent data
            customer_emotion_score=stress_result["stress_score"],
            agent_emotion_score=None,  # Would need separate agent data
        )
        results["empathy_score"] = empathy_result["empathy_score"]
        
        # Step 7: CX score
        logger.debug("Calculating CX score...")
        cx_result = self.cx_calculator.calculate_cx_score(
            stress_score=stress_result["stress_score"],
            empathy_score=empathy_result["empathy_score"],
            silence_ratio=audio_features.get("silence_ratio"),
            silence_duration=audio_features.get("silence_duration"),
            total_duration=audio_features.get("total_duration"),
        )
        results["cx_score"] = cx_result["cx_score"]
        results["cx_breakdown"] = cx_result["breakdown"]
        
        return results
    
    def run(
        self,
        max_samples: Optional[int] = None,
        pattern: str = "*.wav",
        recursive: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Run the complete pipeline on all audio files in the directory.
        
        Args:
            max_samples: Maximum number of samples to process. If None, processes all.
            pattern: File pattern to match. Default is "*.wav".
            recursive: Whether to search recursively. Default is False.
        
        Returns:
            List of result dictionaries for each processed sample.
        """
        logger.info("Starting pipeline execution...")
        
        all_results = []
        samples_processed = 0
        
        for sample in self.data_loader.iter_audio_files(
            pattern=pattern,
            recursive=recursive,
        ):
            if max_samples and samples_processed >= max_samples:
                break
            
            try:
                logger.info(f"Processing call {samples_processed + 1}: {sample.get('call_id')}...")
                result = self.process_sample(sample)
                all_results.append(result)
                samples_processed += 1
                
                if self.print_results:
                    self._print_sample_result(result)
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        logger.info(f"Pipeline completed. Processed {samples_processed} calls.")
        
        # Save results if requested
        if self.save_results and all_results:
            self._save_results(all_results)
        
        return all_results
    
    def _print_sample_result(self, result: Dict[str, any]) -> None:
        """Print sample result to console."""
        print("\n" + "=" * 60)
        print(f"Call ID: {result.get('call_id')} | "
              f"File: {result.get('filename')}")
        print("-" * 60)
        print(f"CX Score: {result.get('cx_score', 0):.1f}/100")
        print(f"  Stress Score: {result.get('stress_score', 0):.3f}")
        print(f"  Empathy Score: {result.get('empathy_score', 0):.3f}")
        print(f"  Silence Ratio: {result.get('audio_features', {}).get('silence_ratio', 0):.2%}")
        print(f"Transcription: {result.get('transcription', {}).get('text', '')[:100]}...")
        print("=" * 60)
    
    def _save_results(self, results: List[Dict[str, any]]) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: List of result dictionaries.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.output_dir / "cx_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json_results = convert_types(results)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Also save summary statistics
        self._save_summary(results)
    
    def _save_summary(self, results: List[Dict[str, any]]) -> None:
        """
        Save summary statistics to CSV.
        
        Args:
            results: List of result dictionaries.
        """
        summary_data = []
        
        for result in results:
            summary_data.append({
                "call_id": result.get("call_id"),
                "filename": result.get("filename"),
                "cx_score": result.get("cx_score", 0),
                "stress_score": result.get("stress_score", 0),
                "empathy_score": result.get("empathy_score", 0),
                "nlp_stress": result.get("nlp_stress", 0),
                "silence_ratio": result.get("audio_features", {}).get("silence_ratio", 0),
                "speech_rate": result.get("audio_features", {}).get("speech_rate", 0),
            })
        
        df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "cx_summary.csv"
        df.to_csv(summary_file, index=False)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print overall statistics
        if self.print_results and len(df) > 0:
            print("\n" + "=" * 60)
            print("OVERALL STATISTICS")
            print("=" * 60)
            print(f"Total Samples: {len(df)}")
            print(f"Average CX Score: {df['cx_score'].mean():.1f}/100")
            print(f"Average Stress Score: {df['stress_score'].mean():.3f}")
            print(f"Average Empathy Score: {df['empathy_score'].mean():.3f}")
            print("=" * 60)


def run_pipeline(
    audio_dir: str | Path = "data/raw/audio/call_center",
    output_dir: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
    whisper_model_size: str = "small",
    save_results: bool = True,
    print_results: bool = True,
    pattern: str = "*.wav",
    recursive: bool = False,
) -> List[Dict[str, any]]:
    """
    Convenience function to run the complete EdgeMindCX pipeline.
    
    Audio-first architecture: processes .wav files directly from directory.
    
    Args:
        audio_dir: Directory containing .wav audio files.
                  Default is "data/raw/audio/call_center".
        output_dir: Directory to save results.
        max_samples: Maximum number of samples to process.
        whisper_model_size: Whisper model size ('small' or 'medium').
        save_results: Whether to save results to JSON.
        print_results: Whether to print results to console.
        pattern: File pattern to match. Default is "*.wav".
        recursive: Whether to search recursively. Default is False.
    
    Returns:
        List of result dictionaries.
    
    Example:
        >>> results = run_pipeline(
        ...     audio_dir="data/raw/audio/call_center",
        ...     max_samples=10
        ... )
    """
    pipeline = EdgeMindCXPipeline(
        audio_dir=audio_dir,
        output_dir=output_dir,
        whisper_model_size=whisper_model_size,
        save_results=save_results,
        print_results=print_results,
    )
    
    return pipeline.run(
        max_samples=max_samples,
        pattern=pattern,
        recursive=recursive,
    )
