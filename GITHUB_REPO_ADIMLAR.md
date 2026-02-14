# GitHub Repo Oluşturma – EdgeMindCX

## ✅ Yapılanlar (bilgisayarında zaten yapıldı)

- Git repo başlatıldı (`git init`)
- Dosyalar eklendi (`git add .`)
- İlk commit atıldı (`Initial commit: EdgeMind CX project`)
- Ana dal `main` yapıldı, remote eklendi: `https://github.com/zeliha-orhan/EdgeMindCX.git`

---

## Adım 1: GitHub'da repo oluştur (sen yapacaksın)

1. Tarayıcıda aç: **https://github.com/new**
2. **Repository name:** `EdgeMindCX` (tam bu isim)
3. **Description:** (isteğe bağlı) `Edge & call center CX analytics – audio, stress, empathy, simulation`
4. **Public** seçili olsun.
5. **"Add a README file"** kutusunu işaretleme (proje zaten yüklenecek).
6. **Create repository** butonuna tıkla.

Repo açıldıktan sonra boş bir sayfa göreceksin; normal, projeyi aşağıdaki push ile göndereceğiz.

---

## Adım 2: Projeyi GitHub'a gönder (PowerShell'de)

Proje klasöründe **PowerShell** açıp tek komutu çalıştır:

```powershell
cd C:\Users\zelih\Desktop\EdgeMindCX
git push -u origin main
```

- İlk kez push’ta Windows veya Git, GitHub girişi isteyebilir (tarayıcı açılır veya pencere çıkar).
- GitHub artık hesap şifresiyle push kabul etmiyor. İki seçenek:
  - **Tarayıcı ile giriş:** Açılan sayfada “Sign in with your browser” deyip GitHub’da giriş yap.
  - **Personal Access Token:** GitHub → Settings → Developer settings → Personal access tokens → “Generate new token (classic)”. `repo` yetkisi ver. Push’ta şifre yerine bu token’ı yapıştır.

Push başarılı olunca `https://github.com/zeliha-orhan/EdgeMindCX` adresinde tüm proje görünür.
