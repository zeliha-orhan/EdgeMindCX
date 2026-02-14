# web/services/file_storage.py
# Yüklenen ses dosyalarının güvenli kaydı. Analiz yapılmaz.

import os
import uuid
from fastapi import UploadFile

ALLOWED_EXTENSION = ".wav"
MAX_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB stream

# Proje köküne göre data/web_uploads (web/services -> web -> kök)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(_PROJECT_ROOT, "data", "web_uploads")


class FileStorageError(Exception):
    """Dosya depolama hataları için temel sınıf."""
    pass


class InvalidFileTypeError(FileStorageError):
    """Sadece .wav kabul edilir."""
    pass


class FileTooLargeError(FileStorageError):
    """Dosya boyutu 200MB sınırını aştı."""
    pass


def _ensure_upload_dir() -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return UPLOAD_DIR


def save_uploaded_audio(upload_file: UploadFile) -> str:
    """
    Yüklenen ses dosyasını doğrular ve data/web_uploads/{call_id}.wav olarak kaydeder.
    Analiz yapılmaz. Hızlı stream ile yazılır.
    Returns:
        call_id: UUID string (dosya adı {call_id}.wav).
    """
    filename = (upload_file.filename or "").strip().lower()
    if not filename.endswith(ALLOWED_EXTENSION):
        raise InvalidFileTypeError(f"Sadece {ALLOWED_EXTENSION} dosyaları kabul edilir. Gönderilen: {filename or '(dosya adı yok)'}")

    call_id = str(uuid.uuid4())
    _ensure_upload_dir()
    path = os.path.join(UPLOAD_DIR, f"{call_id}.wav")

    total = 0
    try:
        with open(path, "wb") as f:
            while True:
                chunk = upload_file.file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_SIZE_BYTES:
                    f.close()
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    raise FileTooLargeError(f"Dosya boyutu 200 MB sınırını aştı (yaklaşık {total // (1024*1024)} MB).")
                f.write(chunk)
    except FileTooLargeError:
        raise
    except FileStorageError:
        raise
    except OSError as e:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise FileStorageError(f"Dosya kaydedilemedi: {e!s}")

    return call_id
