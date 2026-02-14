# EdgeMindCX

Çağrı merkezi ses kayıtlarından davranışsal analiz ve CX skoru üretmek için AI projesi.

## Proje Yapısı

```
EdgeMindCX/
├── edge_mind_cx/          # Ana uygulama paketi
│   ├── audio/             # Ses işleme modülü (yükleme, ön işleme)
│   ├── transcription/     # Speech-to-Text modülü
│   ├── analysis/          # Metin ve ses analizi, özellik çıkarımı
│   ├── behavioral/        # Davranışsal analiz (duygu, ton analizi)
│   ├── scoring/           # CX skoru hesaplama ve metrikler
│   ├── utils/             # Yardımcı fonksiyonlar, logger
│   ├── config/            # Konfigürasyon yönetimi
│   └── api/               # REST API endpoint'leri
│
├── tests/                 # Test dosyaları
│   ├── unit/              # Unit testler
│   └── integration/       # Entegrasyon testleri
│
├── data/                  # Veri klasörleri
│   ├── raw/               # Ham ses kayıtları
│   ├── processed/         # İşlenmiş veriler, transkripsiyonlar
│   └── models/            # Eğitilmiş ML modelleri
│
├── notebooks/             # Jupyter notebook'lar (araştırma, prototipleme)
├── scripts/               # Yardımcı scriptler (batch işlemler)
├── docs/                  # Dokümantasyon
├── config/                # Konfigürasyon dosyaları (YAML, JSON)
├── logs/                  # Log dosyaları
│
├── requirements.txt       # Python bağımlılıkları
├── setup.py              # Proje kurulum dosyası
└── .gitignore            # Git ignore kuralları
```

## Modül Açıklamaları

- **audio/**: Ses dosyalarının yüklenmesi, format dönüşümü, ön işleme
- **transcription/**: Ses kayıtlarının metne dönüştürülmesi
- **analysis/**: Metin ve ses analizi, özellik çıkarımı
- **behavioral/**: Müşteri ve temsilci davranış analizi, duygu/ton analizi
- **scoring/**: CX skoru hesaplama algoritmaları ve metrikler
- **utils/**: Ortak yardımcı fonksiyonlar, veri işleme araçları
- **config/**: Uygulama ve model konfigürasyonları
- **api/**: REST API endpoint'leri (FastAPI/Flask)
