<p align="center">
  <strong style="font-size: 1.2em;">EdgeMind CX</strong>
</p>
<p align="center">
  <em>Çağrı merkezi ses kayıtlarından davranışsal analiz ve CX skoru</em>
</p>
<p align="center">
  <a href="#-özellikler">Özellikler</a> •
  <a href="#-kurulum">Kurulum</a> •
  <a href="#-hızlı-başlangıç">Hızlı Başlangıç</a> •
  <a href="#-proje-yapısı">Proje Yapısı</a> •
  <a href="#-modüller">Modüller</a>
</p>

---

## 🎯 Nedir?

**EdgeMind CX**, çağrı merkezi ses kayıtlarını transkribe eden, stres/empati ve davranışsal metrikleri çıkaran ve **müşteri deneyimi (CX) skoru** üreten bir AI projesidir. Ses verisi edge’de işlenebilir; analiz düşük gecikmeyle yapılır.

---

## ✨ Özellikler

| Özellik | Açıklama |
|--------|----------|
| 🎤 **Speech-to-Text** | Whisper ile ses kayıtlarının metne dönüştürülmesi |
| 📊 **Davranışsal analiz** | Stres, empati ve ton analizi |
| 📈 **CX skoru** | Stres, empati, sessizlik ve akış metriklerinden 0–100 skor |
| 🔀 **Diyarizasyon** | Konuşmacı ayrımı (müşteri / temsilci) |
| 🌐 **Web arayüzü** | Ses yükleme, analiz tetikleme ve simülasyon sayfası (FastAPI) |
| ⚡ **Edge odaklı** | Düşük gecikme, yerelde işleme senaryoları |

---

## 🛠 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/zeliha-orhan/EdgeMindCX.git
cd EdgeMindCX

# Sanal ortam (önerilir)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/macOS

# Bağımlılıkları yükle
pip install -e .
# veya
pip install -r requirements.txt
```

---

## 🚀 Hızlı Başlangıç

**Web arayüzünü çalıştır (port 8001):**

```bash
python -m uvicorn web.main:app --host 127.0.0.1 --port 8001
```

Tarayıcıda:

- **Ana sayfa:** http://127.0.0.1:8001  
- **Simülasyon:** http://127.0.0.1:8001/simulation  

`.wav` dosyası yükleyip analiz başlatabilir; sonuçlar dashboard ve simülasyon sayfasında görünür.

---

## 📁 Proje Yapısı

```
EdgeMindCX/
├── edge_mind_cx/              # Ana uygulama paketi
│   ├── audio/                 # Ses işleme (yükleme, ön işleme)
│   ├── transcription/         # Speech-to-Text (Whisper, diyarizasyon)
│   ├── analysis/              # Metin + ses analizi, özellik çıkarımı
│   ├── behavioral/            # Davranışsal analiz (stres, empati, churn)
│   ├── scoring/               # CX skoru hesaplama ve metrikler
│   ├── utils/                 # Yardımcı fonksiyonlar, logger
│   ├── config/                # Konfigürasyon yönetimi
│   └── api/                   # REST API endpoint'leri
│
├── web/                       # FastAPI web arayüzü
│   ├── main.py                # Uygulama giriş noktası
│   ├── api/                   # Upload, analiz API'leri
│   ├── services/              # Dosya depolama, job yönetimi
│   └── ui/                    # HTML/CSS/JS (ana sayfa + simülasyon)
│
├── tests/                     # Unit ve entegrasyon testleri
├── data/                      # raw/, processed/, models/
├── notebooks/                 # Jupyter (araştırma, prototipleme)
├── scripts/                   # Batch scriptler (transkripsiyon, diyarizasyon, vb.)
├── docs/                      # Dokümantasyon
├── config/                    # YAML/JSON konfigürasyon
├── requirements.txt
├── setup.py
└── run_web.bat                # Windows: web sunucusunu başlatır
```

---

## 📦 Modüller

| Modül | Açıklama |
|-------|----------|
| **audio/** | Ses dosyalarının yüklenmesi, format dönüşümü, ön işleme |
| **transcription/** | Ses kayıtlarının metne dönüştürülmesi (Whisper), konuşmacı ayrımı |
| **analysis/** | Metin ve ses analizi, özellik çıkarımı (openSMILE, librosa) |
| **behavioral/** | Müşteri ve temsilci davranış analizi, duygu/ton, erken churn riski |
| **scoring/** | CX skoru hesaplama algoritmaları ve metrikler |
| **utils/** | Ortak yardımcı fonksiyonlar, veri işleme araçları |
| **config/** | Uygulama ve model konfigürasyonları |
| **api/** | REST API endpoint'leri (FastAPI) |

---

## 📄 Lisans ve Katkı

Proje eğitim ve araştırma amaçlıdır. Soru veya katkı için issue / pull request açabilirsiniz.
