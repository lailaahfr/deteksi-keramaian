# Deteksi Keramaian Lab

Sistem web yang mendengarkan suasana laboratorium lewat mikrofon dan memutar teguran **"sssttt… jangan ramai"** saat lab terdeteksi berisik. Prediksi menggunakan model machine learning hasil klasifikasi audio berisik vs non-berisik.

Proyek Sains Data · Kelompok 12

## Cara Menjalankan

### Dengan Docker

Syarat: [Docker Desktop](https://www.docker.com/products/docker-desktop/) sudah terpasang dan berjalan.

```bash
git clone https://github.com/lailaahfr/deteksi-keramaian.git
cd deteksi-keramaian
docker compose up --build
```

Buka **http://localhost:8012** di browser.

Untuk menghentikan, tekan `Ctrl + C`.

### Tanpa Docker (lokal)

Syarat: [Python 3.13](https://www.python.org/downloads/) sudah terpasang.

```bash
git clone https://github.com/lailaahfr/deteksi-keramaian.git
cd deteksi-keramaian

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python app.py
```

Buka **http://localhost:5000** di browser.

Untuk menghentikan, tekan `Ctrl + C`.
