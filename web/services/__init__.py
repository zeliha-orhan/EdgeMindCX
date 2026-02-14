# web/services/
# ---------------------------------------------------------------------------
# İş mantığı katmanı — edge_mind_cx ile köprü, veri dönüşümü, orkestrasyon.
#
# İçermeli:
#   - edge_mind_cx modüllerini çağıran servisler
#   - Veri dönüşümü (analiz çıktısı → API'ye uygun format)
#   - Dış servis/veritabanı erişimi (gerekirse)
#
# İçermemeli:
#   - HTTP/route detayları (api/ tarafında)
#   - Ham analiz algoritmaları (proje kökündeki analiz koduna dokunma)
# ---------------------------------------------------------------------------
