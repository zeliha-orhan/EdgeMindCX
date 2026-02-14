# web/api/analyze.py
# Job tabanlı analiz: POST /api/analyze/{call_id} -> 202, GET /api/job/{job_id} -> durum.

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from web.services.file_storage import UPLOAD_DIR
from web.services.job_storage import create_job, get_job
from web.services.analysis_runner import run_analysis

router = APIRouter()


def _wav_path(call_id: str) -> Path:
    return Path(UPLOAD_DIR) / f"{call_id}.wav"


@router.post(
    "/analyze/{call_id}",
    status_code=202,
    summary="Analiz job'ı başlat",
    responses={
        202: {"description": "Job kabul edildi, arka planda çalışıyor"},
        400: {"description": "Geçersiz call_id"},
        404: {"description": "call_id için yüklenmiş ses dosyası yok"},
    },
)
async def start_analyze(call_id: str, background_tasks: BackgroundTasks):
    """
    Anında 202 Accepted döner. Analiz arka planda çalışır; request bloklanmaz.
    Durum: GET /api/jobs/{job_id} ile sorgulanır.
    """
    if "/" in call_id or "\\" in call_id or ".." in call_id:
        raise HTTPException(status_code=400, detail="Geçersiz call_id")
    if not _wav_path(call_id).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Yüklenmiş ses dosyası bulunamadı: call_id={call_id}",
        )
    job_id = str(uuid.uuid4())
    create_job(job_id, call_id)

    audio_path = _wav_path(call_id)
    background_tasks.add_task(_run_analysis_in_executor, job_id, audio_path)
    return {"job_id": job_id, "status": "accepted"}


async def _run_analysis_in_executor(job_id: str, audio_path: Path) -> None:
    """Analizi thread pool'da çalıştırır; event loop bloklanmaz."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_analysis, job_id, audio_path)


def _elapsed_seconds(job: dict) -> float:
    """created_at ile updated_at (yoksa şimdi) arası saniye."""
    created = job.get("created_at")
    if not created:
        return 0.0
    end = job.get("updated_at") or datetime.now(tz=timezone.utc).isoformat()
    try:
        t0 = datetime.fromisoformat(created.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0.0, (t1 - t0).total_seconds())
    except (ValueError, TypeError):
        return 0.0


@router.get(
    "/job/{job_id}",
    summary="Job durumunu sorgula (status, progress, elapsed)",
    responses={
        200: {"description": "Job durum özeti"},
        404: {"description": "Job bulunamadı"},
    },
)
async def get_job_status(job_id: str):
    """
    Hızlı, bloklamaz. status, progress (0-100), current_step, elapsed_time (saniye) döner.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job bulunamadı: job_id={job_id}")
    progress = job.get("progress", 0)
    if not isinstance(progress, (int, float)):
        progress = 0
    progress = max(0, min(100, int(progress)))
    out = {
        "status": job.get("status", "unknown"),
        "progress": progress,
        "current_step": job.get("current_step", ""),
        "elapsed_time": _elapsed_seconds(job),
    }
    if job.get("status") == "error" and job.get("error"):
        out["error"] = job["error"]
    return out


@router.get(
    "/jobs/{job_id}",
    summary="Job ham verisi (tüm alanlar)",
    responses={
        200: {"description": "Job JSON (result/error dahil)"},
        404: {"description": "Job bulunamadı"},
    },
)
async def get_job_full(job_id: str):
    """data/jobs/{job_id}.json tam içeriği (result, error vb.)."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job bulunamadı: job_id={job_id}")
    return job
