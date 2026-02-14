# web/services/analysis_runner.py
# Mevcut full analiz pipeline'ını tek fonksiyonla çağırır.
# Ses dosyası path alır; job durumunu günceller. Analiz koduna dokunulmaz.
# Pipeline/torch import'u sadece run_analysis içinde yapılır (lazy) böylece sunucu
# torch/torchcodec yüklemeden ayağa kalkar; tarayıcı arayüze bağlanabilir.

import logging
from pathlib import Path
from typing import Union

from web.services.job_storage import (
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    update_job_status,
)

logger = logging.getLogger(__name__)


def _convert_for_json(obj):
    """numpy ve diğer JSON'a uyumsuz tipleri dönüştürür."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_analysis(job_id: str, audio_path: Union[str, Path]) -> None:
    """
    Tek fonksiyon: ses dosyası path'i alır, mevcut full pipeline'ı çalıştırır,
    sonucu job durumuna yazar. Analiz koduna dokunulmaz; web sadece sonucu alır.

    Thread/executor içinde çağrılmalı (request bloklanmasın).

    Args:
        job_id: Job kimliği (durum bu id ile güncellenir).
        audio_path: .wav dosyasının tam yolu.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        update_job_status(job_id, JOB_STATUS_ERROR, error=f"Dosya bulunamadı: {path}")
        return
    if path.suffix.lower() != ".wav":
        update_job_status(job_id, JOB_STATUS_ERROR, error="Sadece .wav dosyaları desteklenir")
        return
    audio_dir = str(path.parent)
    call_id = path.stem
    try:
        # Lazy import: torch/whisper/pyannote sadece analiz çalışırken yüklensin;
        # sunucu başlarken libtorchcodec hatası vermesin, tarayıcı bağlanabilsin.
        from edge_mind_cx.audio_loader import AudioDataLoader
        from edge_mind_cx.pipeline import EdgeMindCXPipeline

        loader = AudioDataLoader(audio_dir=audio_dir)
        sample = loader.load_audio_file(path, call_id=call_id)
        pipeline = EdgeMindCXPipeline(
            audio_dir=audio_dir,
            output_dir=None,
            save_results=False,
            print_results=False,
        )
        result = pipeline.process_sample(sample)
        update_job_status(
            job_id,
            JOB_STATUS_DONE,
            result=_convert_for_json(result),
            progress=100,
            current_step="done",
        )
        logger.info("Analysis completed job_id=%s path=%s", job_id, path)
    except Exception as e:
        logger.exception("Analysis failed job_id=%s path=%s", job_id, path)
        update_job_status(
            job_id,
            JOB_STATUS_ERROR,
            error=str(e),
            current_step="error",
        )
