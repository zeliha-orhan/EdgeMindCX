# web/main.py
# ---------------------------------------------------------------------------
# Web arayüzünün giriş noktası.
#
# İçermeli:
#   - Uygulama örneği (FastAPI/Flask vb.)
#   - Web sunucusunun başlatılması
#   - Ana rotaların mount edilmesi
#
# İçermemeli:
#   - Analiz/veri işleme mantığı (edge_mind_cx/ tarafında kalmalı)
#   - Doğrudan veritabanı veya dosya erişimi (services üzerinden)
# ---------------------------------------------------------------------------

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from web.api.upload import router as upload_router
from web.api.analyze import router as analyze_router
from web.services.file_storage import UPLOAD_DIR
from web.services.job_storage import JOBS_DIR

app = FastAPI(title="EdgeMind CX Web")

# POST /api/upload — ses dosyası yükleme
app.include_router(upload_router, prefix="/api", tags=["upload"])
# POST /api/analyze/{call_id} — job başlat, GET /api/jobs/{job_id} — job durumu
app.include_router(analyze_router, prefix="/api", tags=["analyze"])


@app.on_event("startup")
def _ensure_dirs():
    import os
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(JOBS_DIR, exist_ok=True)


# Basit UI: web/ui/index.html
_UI_INDEX = Path(__file__).resolve().parent / "ui" / "index.html"
# Simulation sayfası: web/ui/simulation/simulation.html
_SIMULATION_INDEX = Path(__file__).resolve().parent / "ui" / "simulation" / "simulation.html"


@app.get("/", include_in_schema=False)
def serve_ui():
    """Ana sayfa: basit HTML arayüzü (upload, analiz, job polling)."""
    if _UI_INDEX.exists():
        return FileResponse(_UI_INDEX, media_type="text/html")
    return {"message": "UI not found"}


@app.get("/simulation", include_in_schema=False)
def serve_simulation():
    """Simulation sonuçları sayfası."""
    if _SIMULATION_INDEX.exists():
        return FileResponse(_SIMULATION_INDEX, media_type="text/html")
    return {"message": "Simulation UI not found"}


@app.get("/health")
def health():
    """Servis ayakta mı kontrolü. Dosya upload / analiz yok."""
    return {"status": "ok", "service": "web"}
