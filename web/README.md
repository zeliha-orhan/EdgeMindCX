# EdgeMind CX Web

## Sunucuyu başlatma

Proje kök dizininde (EdgeMindCX) şu komutu çalıştırın:

```bash
uvicorn web.main:app --host 127.0.0.1 --port 8001
```

veya:

```bash
python -m uvicorn web.main:app --host 127.0.0.1 --port 8001
```

## Tarayıcıda açma

Sunucu çalıştıktan sonra tarayıcıda şu adresi açın:

- **http://127.0.0.1:8001**

`localhost` kullanıyorsanız:

- **http://localhost:8001**

Port 8001 kullanılıyorsa farklı bir port deneyin: `--port 8080`

## libtorchcodec / FFmpeg uyarısı

Sunucu başlarken terminalde "Could not load libtorchcodec" veya FFmpeg ile ilgili uyarılar görebilirsiniz. Bu uyarılar sunucunun çalışmasını engellemez; arayüz açılır. İlk analiz başlatıldığında bu kütüphaneler yüklenecektir. İsterseniz [FFmpeg](https://ffmpeg.org/download.html) kurarak uyarıyı azaltabilirsiniz.
