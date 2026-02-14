# EdgeMind CX – Teknik Mimari

## Sistem mimarisi (özet akış)

```mermaid
flowchart LR
  subgraph Kullanıcı
    UI[Web UI]
  end
  
  subgraph Sunucu
    API[FastAPI]
  end
  
  subgraph AI Katmanı
    PIPE[EdgeMind CX Pipeline]
  end
  
  subgraph Simülasyon
    SIM[Edge Simulation]
  end
  
  UI -->|Upload / Analiz isteği| API
  API -->|Ses işleme| PIPE
  PIPE -->|Transkript, skorlar| API
  API -->|Sonuçlar, job durumu| UI
  SIM -->|Metrikler, senaryolar| UI
```

**Açıklama:** Tarayıcıdaki Web UI, ses yükleme ve analiz isteklerini FastAPI’ye gönderir. FastAPI, EdgeMind CX pipeline’ını (Whisper, özellik çıkarımı, davranışsal analiz, CX skoru) çalıştırır; sonuçlar API üzerinden UI’a döner. Edge Simulation sayfası, aynı UI içinde simüle edge/cloud metriklerini ve senaryoları gösterir.

---

## Detaylı bileşen diyagramı

```mermaid
flowchart TB
  subgraph "Tarayıcı"
    INDEX["Ana sayfa (index.html)"]
    SIMP["Simülasyon (simulation.html)"]
  end

  subgraph "FastAPI (web/)"
    MAIN["main.py"]
    UPLOAD["/api/upload"]
    ANALYZE["/api/analyze"]
    SERVES["/ , /simulation"]
  end

  subgraph "EdgeMind CX Pipeline (edge_mind_cx/)"
    LOAD["Audio Loader"]
    WHISPER["Whisper (STT)"]
    FEAT["Özellik çıkarımı"]
    BEH["Davranışsal analiz"]
    SCORE["CX skoru"]
  end

  subgraph "Veri / İşler"
    FILES["Dosya depolama"]
    JOBS["Job durumu (JSON)"]
  end

  INDEX --> UPLOAD
  INDEX --> ANALYZE
  MAIN --> SERVES
  SERVES --> INDEX
  SERVES --> SIMP
  UPLOAD --> FILES
  ANALYZE --> LOAD
  LOAD --> WHISPER
  WHISPER --> FEAT
  FEAT --> BEH
  BEH --> SCORE
  SCORE --> JOBS
  ANALYZE --> JOBS
  INDEX -.->|polling| JOBS
  SIMP -.->|frontend only| SIMP
```

---

## Bileşenler

| Bileşen | Konum | Görevi |
|--------|--------|--------|
| **Web UI** | `web/ui/` | Ses yükleme, analiz tetikleme, dashboard, simülasyon sayfası |
| **FastAPI** | `web/main.py`, `web/api/` | REST API, HTML servisi, upload/analyze endpoint’leri |
| **AI Pipeline** | `edge_mind_cx/` | Ses → Whisper → özellikler → davranışsal analiz → CX skoru |
| **Edge Simulation** | `web/ui/simulation/` | What-if senaryoları, edge/cloud dağılımı (sadece frontend) |

---

## Veri akışı (analiz isteği)

1. Kullanıcı `.wav` yükler → **FastAPI** `/api/upload` ile dosyayı kaydeder.
2. Kullanıcı analiz başlatır → **FastAPI** `/api/analyze/{call_id}` job oluşturur; arka planda **EdgeMind CX pipeline** çalışır.
3. Pipeline: ses → Whisper (transkript) → özellikler → stres/empati → CX skoru → sonuç **job JSON** olarak saklanır.
4. UI job id ile durum sorgular; tamamlanınca sonuçları (transkript, skorlar) gösterir.
5. **Simülasyon** sayfası sabit/simüle metriklerle edge vs cloud karşılaştırması yapar; backend’e istek atmaz.
