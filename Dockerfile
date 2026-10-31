# ---------------------------------------------------------------
# Deteksi Keramaian Lab ("sssttt... jangan ramai") - Kelompok 12
# Build : docker build -t deteksi-keramaian .
# Run   : docker run -p 8012:5000 deteksi-keramaian
# ---------------------------------------------------------------

# Python 3.13 = versi yang dipakai saat melatih model (lihat paket["versi"] di model)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# libsndfile: dipakai soundfile/librosa untuk membaca file audio (mp3, wav, flac, ogg)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependensi dulu supaya layer ini di-cache selama requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin kode aplikasi, model, halaman web, dan suara teguran
COPY . .

EXPOSE 5000

# Cek kesehatan: aplikasi dianggap sehat jika /api/info bisa diakses
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/info', timeout=4)"

# Cara menjalankan aplikasi.
# --host 0.0.0.0 wajib agar aplikasi bisa diakses dari luar container.
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
