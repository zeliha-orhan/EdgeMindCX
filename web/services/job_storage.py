# web/services/job_storage.py
# Job durumlarının data/jobs/{job_id}.json ile saklanması.

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

# Proje köküne göre data/jobs
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS_DIR = os.path.join(_PROJECT_ROOT, "data", "jobs")

JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"


def _ensure_jobs_dir() -> str:
    os.makedirs(JOBS_DIR, exist_ok=True)
    return JOBS_DIR


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def create_job(job_id: str, call_id: str) -> None:
    """Job oluşturur, durumu 'running', progress=0, current_step='running' yazar."""
    _ensure_jobs_dir()
    path = _job_path(job_id)
    payload = {
        "job_id": job_id,
        "call_id": call_id,
        "status": JOB_STATUS_RUNNING,
        "progress": 0,
        "current_step": "running",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def update_job_status(
    job_id: str,
    status: str,
    *,
    error: Optional[str] = None,
    result: Optional[dict[str, Any]] = None,
    progress: Optional[int] = None,
    current_step: Optional[str] = None,
) -> None:
    """Job durumunu günceller (done veya error). progress/current_step opsiyonel."""
    path = _job_path(job_id)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["status"] = status
    payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    if error is not None:
        payload["error"] = error
    if result is not None:
        payload["result"] = result
    if progress is not None:
        payload["progress"] = max(0, min(100, progress))
    if current_step is not None:
        payload["current_step"] = current_step
    if status == JOB_STATUS_DONE and progress is None:
        payload.setdefault("progress", 100)
        payload.setdefault("current_step", "done")
    if status == JOB_STATUS_ERROR and current_step is None:
        payload.setdefault("current_step", "error")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Job bilgisini okur. Yoksa None döner."""
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
