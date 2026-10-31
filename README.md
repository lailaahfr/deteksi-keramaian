# Penjaga Tenang Lab ("sssttt… jangan ramai")

Klasifikasi audio **berisik vs tidak berisik** (Librosa + scikit-learn) dengan alur CRISP-DM,
lalu model terbaik dipakai **sistem web** di lab yang memutar teguran suara saat lab ramai.

## Isi folder
| File | Fungsi |
|---|---|
| `01_klasifikasi_keramaian_lab.ipynb` | Notebook CRISP-DM: Business Understanding → Deployment, membandingkan Random Forest, SVM, KNN |
| `fitur_audio.py` | Pipeline pra-pemrosesan & 32 fitur. Dipakai notebook **dan** sistem agar fitur selalu identik |
| `app.py` | **Sistem web** (Flask): menerima audio mikrofon dari browser, memprediksi, memberi perintah teguran |
| `templates/index.html`, `static/` | Halaman web: Pantau, Uji File Audio, Model (tanpa CDN, bisa offline) |
| `deteksi_lab.py` | Logika keputusan (dipakai `app.py`) + versi terminal/CLI cadangan |
| `buat_suara_peringatan.py` | Membuat `suara/peringatan.wav` (TTS atau rekam suara sendiri) |
| `data/berisik/`, `data/non_berisik/` | Taruh 100 + 100 file audio di sini (MP3 didukung, juga wav/flac/ogg) |
| `Dockerfile`, `docker-compose.yml` | Deployment dengan Docker (port 8012) |
| `model/`, `hasil/`, `suara/` | Diisi otomatis: model, grafik/CSV evaluasi & log deteksi, file suara |

## Menjalankan dengan Docker (Kelompok 12, port 8012)
Syarat: Docker Desktop sudah terpasang dan berjalan.
```bash
git clone https://github.com/lailaahfr/deteksi-keramaian.git
cd deteksi-keramaian
docker compose up -d --build
```
Buka **http://localhost:8012**, lalu tab Pantau, lalu "Mulai mendengarkan".

| Perintah | Fungsi |
|---|---|
| `docker compose up -d --build` | Build image dan jalankan container di background |
| `docker compose ps` | Cek status container |
| `docker compose logs -f` | Lihat log aplikasi |
| `docker compose down` | Hentikan container |

Tanpa Compose: `docker build -t deteksi-keramaian .` lalu `docker run -p 8012:5000 deteksi-keramaian`.

File deployment di root proyek:
- `Dockerfile` berisi resep image dan cara menjalankan aplikasi (`python app.py --host 0.0.0.0 --port 5000`).
- `docker-compose.yml` memetakan port 8012 (laptop) ke 5000 (container). Log deteksi tersimpan di `hasil/`.
- `requirements.txt` berisi dependensi sistem web. Dependensi notebook ada di `requirements-notebook.txt`.

## Langkah (tanpa Docker)
```bash
pip install -r requirements.txt -r requirements-notebook.txt

# 1. Training & evaluasi: buka notebook, Run All
jupyter notebook 01_klasifikasi_keramaian_lab.ipynb
#    -> menghasilkan model/model_terbaik.joblib dan grafik di hasil/

# 2. (opsional) suara teguran sendiri. Tanpa file ini, web memakai desis + suara browser "jangan ramai"
python buat_suara_peringatan.py --rekam 3   # rekam suara sendiri 3 detik (paling natural)
python buat_suara_peringatan.py             # atau desis + Google TTS (butuh internet)

# 3. Jalankan sistem web di PC lab
python app.py
#    buka http://localhost:5000 -> tab Pantau -> "Mulai mendengarkan" -> izinkan mikrofon
```

## Halaman web
- **Pantau**: status besar TENANG / BERISIK, probabilitas, level mikrofon, grafik 3 menit, statistik,
  riwayat teguran. Pengaturan bisa diubah langsung dan tersimpan di browser:
  | Pengaturan | Default | Arti |
  |---|---|---|
  | Mikrofon | default sistem | Pilih mic yang terpasang di lab |
  | Ambang berisik | dari model | Naikkan jika teguran terlalu sering |
  | Tegur jika … dari penilaian | 3 dari 4 | Abaikan suara sesaat (kursi jatuh, tertawa sebentar) |
  | Jeda antar teguran | 30 detik | Supaya tidak menegur terus-menerus |
  | Volume teguran | 100% | Pengali volume suara teguran |
- **Uji File Audio**: unggah MP3/WAV, lihat penilaian per potongan 5 detik dan simulasi kapan sistem menegur.
- **Model**: metode terpilih dan tabel perbandingan 3 metode dari notebook.

Cara kerja: browser merekam mic dan mengirim audio tiap 1 detik → server menilai **jendela 5 detik
terakhir** dengan fitur yang sama persis dengan training → jika 3 dari 4 penilaian berisik, halaman
menampilkan layar "sssttt… jangan ramai" dan memutar suaranya. Selama teguran diputar, audio diabaikan
agar sistem tidak mendeteksi suaranya sendiri. Semua penilaian dicatat di `hasil/log_deteksi.csv`.

## Catatan penting
- **Buka lewat `http://localhost:5000` di PC lab itu sendiri.** Browser hanya mengizinkan mikrofon di
  `localhost` atau HTTPS. Jika dibuka dari PC lain (`python app.py --host 0.0.0.0` lalu `http://IP:5000`),
  mikrofon akan diblokir kecuali memakai HTTPS.
- Biarkan tab tetap terbuka dan speaker menyala. Halaman meminta layar tetap menyala (Wake Lock) bila browser mendukung.
- Browser diminta **tidak** memakai noise suppression / auto gain, supaya volume asli lab tidak diubah
  (fitur RMS bergantung pada volume).
- **Split per file**: data uji dipisah per rekaman asli sebelum chunking, dan cross-validation memakai
  `StratifiedGroupKFold`, sehingga chunk dari satu rekaman tidak bocor ke data uji.
- **32 fitur**, bukan 40 seperti data dummy di modul: 2 RMS + 2 centroid + 2 ZCR + 13 mean MFCC + 13 std MFCC.
- **Mikrofon**: hasil terbaik didapat bila data latih direkam dengan mikrofon dan posisi yang sama dengan
  sistem di lab. Jika terlalu sensitif atau kurang sensitif, atur ambang di halaman web, atau rekam
  beberapa sampel tambahan dari mikrofon lab lalu latih ulang.
- Jika `fitur_audio.py` diubah (misal durasi chunk 10 detik), jalankan ulang notebook. Sistem menolak
  model lama yang konfigurasinya berbeda.
- **File MP3** dibaca librosa lewat `soundfile` ≥ 0.12.1. Jika muncul error saat membaca MP3, jalankan
  `pip install -U soundfile` atau instal ffmpeg. Notebook punya sel pengecekan di awal Data Understanding.
- Cadangan tanpa browser: `python deteksi_lab.py` (butuh `sounddevice`).
