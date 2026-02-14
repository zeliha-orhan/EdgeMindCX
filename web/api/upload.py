# web/api/upload.py
# Ses dosyası yükleme endpoint'i. Analiz yapılmaz.

from fastapi import APIRouter, File, HTTPException, UploadFile

from web.services.file_storage import (
    FileStorageError,
    FileTooLargeError,
    InvalidFileTypeError,
    save_uploaded_audio,
)

router = APIRouter()


@router.post(
    "/upload",
    summary="Ses dosyası yükle",
    response_model=None,
    responses={
        200: {"description": "Dosya başarıyla kaydedildi"},
        400: {"description": "Geçersiz dosya türü (sadece .wav)"},
        413: {"description": "Dosya 200 MB sınırını aştı"},
        500: {"description": "Sunucu hatası"},
    },
)
async def upload_audio(
    file: UploadFile = File(..., description=".wav ses dosyası (maks. 200 MB)"),
):
    """
    Sadece .wav kabul edilir, maksimum 200 MB. Dosya data/web_uploads/{call_id}.wav olarak kaydedilir.
    Analiz çalıştırılmaz.
    """
    try:
        call_id = save_uploaded_audio(file)
        return {"call_id": call_id, "status": "uploaded"}
    except InvalidFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FileStorageError as e:
        raise HTTPException(status_code=500, detail=str(e))
