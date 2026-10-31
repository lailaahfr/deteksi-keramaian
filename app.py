"""
app.py - Sistem web deteksi keramaian lab
=========================================
Memakai model terbaik dari notebook `klasifikasi-audio.ipynb`
(file model/model_terbaik.joblib yang dibuat di Bab 9 notebook).

Menjalankan:
    python app.py               # buka http://localhost:5000 di PC lab
    python app.py --dev         # mode pengembangan (restart otomatis saat file diubah)
    gunicorn app:app --workers 1 --threads 8 --bind 0.0.0.0:$PORT   # hosting online

Alur mode Pantau:
    mic browser (sample rate asli, PCM 16-bit, tiap 1 detik) -> buffer per sesi
    -> ambil jendela N detik terakhir (default 5) -> resample 16 kHz polifase
    -> pipeline notebook (chunk 5 dtk, buang hening -80 dBFS, 32 fitur mean/std)
    -> model.predict_proba -> aturan "3 dari 4" + jeda -> teguran
"""

import argparse
import collections
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

import fitur_audio as fa
from detektor import Detektor, Pencatat, muat_model, nilai_sinyal, prob_berisik

LOKASI = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(LOKASI, "model", "model_terbaik.joblib")
PATH_SUARA = os.path.join(LOKASI, "suara", "peringatan.wav")
PATH_LOG = os.path.join(LOKASI, "hasil", "log_deteksi.csv")
PATH_CONTOH = os.path.join(LOKASI, "contoh")
EKSTENSI_AUDIO = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
PILIHAN_JENDELA = (5, 10, 15, 20, 30)     # kelipatan 5 detik (panjang chunk notebook)
JENDELA_MAKS = max(PILIHAN_JENDELA)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

PAKET = None
PENCATAT = None
KUNCI_LOG = threading.Lock()
RIWAYAT_TEGURAN = collections.deque(maxlen=200)
SESI = {}
KUNCI_SESI = threading.Lock()


# ----------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------
def bersihkan(obj):
    """Ubah tipe numpy menjadi tipe Python biasa agar bisa di-JSON-kan."""
    if isinstance(obj, dict):
        return {str(k): bersihkan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [bersihkan(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def angka(nilai, bawaan, tipe=float, minimum=None, maksimum=None):
    try:
        x = tipe(nilai)
    except (TypeError, ValueError):
        return bawaan
    if minimum is not None:
        x = max(minimum, x)
    if maksimum is not None:
        x = min(maksimum, x)
    return x


def jendela_valid(nilai, bawaan=fa.DURASI_CHUNK):
    j = angka(nilai, bawaan, int)
    return j if j in PILIHAN_JENDELA else bawaan


def ambang_bawaan():
    return PAKET.get("ambang_proba", 0.5)


class Sesi:
    """Satu tab browser yang sedang mendengarkan mikrofon."""

    def __init__(self, pengaturan):
        self.kunci = threading.Lock()
        self.buffer = np.zeros(0, dtype=np.float32)   # sample rate asli mikrofon
        self.sr = None
        self.jendela = fa.DURASI_CHUNK
        self.terakhir_aktif = time.time()
        self.detektor = Detektor(PAKET, ambang_bawaan(), 3, 4, 30.0)
        self.atur(pengaturan)

    def atur(self, p):
        d = self.detektor
        d.ambang = angka(p.get("ambang"), d.ambang, float, 0.05, 0.99)
        d.cooldown = angka(p.get("cooldown"), d.cooldown, float, 0, 3600)
        riwayat = angka(p.get("riwayat"), d.riwayat.maxlen, int, 1, 10)
        d.konfirmasi = angka(p.get("konfirmasi"), d.konfirmasi, int, 1, riwayat)
        if riwayat != d.riwayat.maxlen:
            d.riwayat = collections.deque(d.riwayat, maxlen=riwayat)
        self.jendela = jendela_valid(p.get("jendela"), self.jendela)

    def pengaturan(self):
        d = self.detektor
        return dict(ambang=d.ambang, cooldown=d.cooldown, konfirmasi=d.konfirmasi,
                    riwayat=d.riwayat.maxlen, jendela=self.jendela)


def ambil_sesi(sid):
    with KUNCI_SESI:
        batas = time.time() - 600                      # sesi ditinggal > 10 menit dihapus
        for k in [k for k, v in SESI.items() if v.terakhir_aktif < batas]:
            SESI.pop(k, None)
        s = SESI.get(sid)
    if s is None:
        abort(404, description="Sesi tidak ditemukan, muat ulang halaman.")
    s.terakhir_aktif = time.time()
    return s


def catat(prob, status, tegur, sumber="mikrofon"):
    with KUNCI_LOG:
        PENCATAT.tulis(prob, status, tegur)
        if tegur:
            RIWAYAT_TEGURAN.appendleft({"waktu": datetime.now().isoformat(timespec="seconds"),
                                        "prob": round(prob, 3), "sumber": sumber})


# ----------------------------------------------------------------------
# Halaman & info
# ----------------------------------------------------------------------
@app.get("/")
def beranda():
    return render_template("index.html")


@app.get("/api/info")
def info():
    return jsonify(bersihkan({
        "metode": PAKET["nama_metode"],
        "parameter": PAKET.get("parameter"),
        "kelas": PAKET.get("kelas"),
        "ambang_bawaan": ambang_bawaan(),
        "dibuat": PAKET.get("dibuat"),
        "metrik_uji": PAKET.get("metrik_uji", {}),
        "perbandingan": PAKET.get("perbandingan", []),
        # list pasangan [fitur, nilai] agar urutan (terbesar dulu) tidak diubah jsonify
        "feature_importance": list(PAKET.get("feature_importance", {}).items()),
        "jumlah_data": PAKET.get("jumlah_data", {}),
        "konfigurasi": {k: v for k, v in PAKET["konfigurasi_fitur"].items() if k != "urutan_fitur"},
        "jumlah_fitur": len(PAKET["fitur"]),
        "pilihan_jendela": PILIHAN_JENDELA,
        "ada_suara": os.path.exists(PATH_SUARA),
        "contoh": daftar_contoh(),
        "versi": PAKET.get("versi", {}),
    }))


@app.get("/suara/peringatan.wav")
def suara():
    if not os.path.exists(PATH_SUARA):
        abort(404)
    return send_file(PATH_SUARA, mimetype="audio/wav", max_age=0)


@app.get("/api/riwayat")
def riwayat():
    return jsonify(list(RIWAYAT_TEGURAN)[:50])


@app.get("/health")
def health():
    """Dipakai layanan pemantau (mis. UptimeRobot) agar server gratis tidak tertidur."""
    return jsonify({"status": "ok", "model": PAKET["nama_metode"] if PAKET else None})


# ----------------------------------------------------------------------
# Mode Pantau: mikrofon browser
# ----------------------------------------------------------------------
@app.post("/api/sesi")
def buat_sesi():
    sid = uuid.uuid4().hex
    s = Sesi(request.get_json(silent=True) or {})
    with KUNCI_SESI:
        SESI[sid] = s
    return jsonify({"sesi": sid, "pengaturan": s.pengaturan()})


@app.post("/api/sesi/<sid>/pengaturan")
def ubah_pengaturan(sid):
    s = ambil_sesi(sid)
    with s.kunci:
        s.atur(request.get_json(silent=True) or {})
        return jsonify(s.pengaturan())


@app.post("/api/sesi/<sid>/reset")
def reset_sesi(sid):
    """Dipanggil setelah teguran selesai diputar: buang audio yang berisi suara teguran."""
    s = ambil_sesi(sid)
    with s.kunci:
        s.buffer = np.zeros(0, dtype=np.float32)
    return jsonify({"ok": True})


@app.post("/api/sesi/<sid>/audio")
def terima_audio(sid):
    """Body: PCM int16 little-endian mono. Query: sr = sample rate mikrofon."""
    s = ambil_sesi(sid)
    sr = angka(request.args.get("sr"), 0, int)
    if not 8000 <= sr <= 192000:
        abort(400, description="Parameter sr tidak valid")
    blok = np.frombuffer(request.get_data(), dtype="<i2").astype(np.float32) / 32768.0

    with s.kunci:
        if s.sr != sr:                                  # mikrofon berganti -> mulai ulang
            s.sr, s.buffer = sr, np.zeros(0, dtype=np.float32)
        s.buffer = np.concatenate([s.buffer, blok])[-int(JENDELA_MAKS * sr):]
        n = int(s.jendela * sr)
        if len(s.buffer) < n:
            return jsonify({"siap": False, "terkumpul": round(len(s.buffer) / sr, 1),
                            "dibutuhkan": s.jendela})
        target = s.jendela * fa.SR
        sinyal = fa.resample(s.buffer[-n:], sr)[:target]
        if len(sinyal) < target:                        # jaga-jaga selisih pembulatan resample
            sinyal = np.pad(sinyal, (0, target - len(sinyal)))
        t0 = time.perf_counter()
        prob, status, tegur = s.detektor.nilai(sinyal, time.time())
        waktu_ms = (time.perf_counter() - t0) * 1000
        jml_berisik, jml_riwayat = int(sum(s.detektor.riwayat)), len(s.detektor.riwayat)
        ambang, jendela = s.detektor.ambang, s.jendela

    catat(prob, status, tegur)
    level_db = float(20 * np.log10(np.sqrt(np.mean(sinyal[-fa.SR:] ** 2)) + 1e-10))
    return jsonify({"siap": True, "prob": round(prob, 4), "status": status, "tegur": bool(tegur),
                    "ambang": ambang, "jendela": jendela, "level_db": round(level_db, 1),
                    "riwayat_berisik": jml_berisik, "riwayat_isi": jml_riwayat,
                    "waktu_ms": round(waktu_ms, 1)})


# ----------------------------------------------------------------------
# Uji file audio
# ----------------------------------------------------------------------
def daftar_contoh():
    """File audio di folder contoh/ untuk dicoba pengunjung (mis. dosen)."""
    if not os.path.isdir(PATH_CONTOH):
        return []
    return sorted(f for f in os.listdir(PATH_CONTOH) if f.lower().endswith(EKSTENSI_AUDIO))


def analisis(sinyal, nama, ambang, jendela):
    durasi = len(sinyal) / fa.SR

    # (a) satu file -> satu prediksi, PERSIS seperti ekstrak_file + model di notebook
    fitur_file = fa.ekstrak_sinyal(sinyal)
    if fitur_file is None:
        prob_file, label_file, jml_chunk = 0.0, "hening", 0
    else:
        prob_file = prob_berisik(PAKET, fitur_file)
        label_file = "BERISIK" if prob_file >= ambang else "TIDAK BERISIK"
        jml_chunk = fitur_file["jumlah_chunk"]

    # (b) penilaian tiap chunk 5 detik
    potongan = []
    for i, chunk in enumerate(fa.potong_chunk(sinyal)):
        p, status = nilai_sinyal(PAKET, chunk)
        if status is None:
            status = "BERISIK" if p >= ambang else "tenang"
        potongan.append({"mulai": i * fa.DURASI_CHUNK, "selesai": (i + 1) * fa.DURASI_CHUNK,
                         "prob": round(p, 4) if status != "hening" else None, "status": status})

    # (c) simulasi sistem: jendela `jendela` detik digeser tiap `langkah` detik + aturan 3 dari 4
    n = int(jendela * fa.SR)
    sinyal_sim = np.pad(sinyal, (0, max(0, n - len(sinyal))))
    langkah = max(1.0, durasi / 300)
    det = Detektor(PAKET, ambang, 3, 4, 30.0)
    simulasi = []
    for awal in range(0, len(sinyal_sim) - n + 1, int(langkah * fa.SR)):
        t = (awal + n) / fa.SR
        p, status, tegur = det.nilai(sinyal_sim[awal:awal + n], t)
        simulasi.append({"t": round(t, 1), "prob": round(p, 4), "tegur": bool(tegur)})

    return {
        "nama": nama, "durasi": round(durasi, 1), "ambang": ambang, "jendela": jendela,
        "prob_file": round(prob_file, 4), "keputusan": label_file, "jumlah_chunk": jml_chunk,
        "potongan": potongan, "simulasi": simulasi,
        "jumlah_teguran": int(sum(s["tegur"] for s in simulasi)),
    }


def parameter_uji(sumber):
    return (angka(sumber.get("ambang"), ambang_bawaan(), float, 0.05, 0.99),
            jendela_valid(sumber.get("jendela")))


@app.post("/api/uji-file")
def uji_file():
    f = request.files.get("file")
    if f is None or not f.filename:
        abort(400, description="Tidak ada file yang diunggah")
    akhiran = os.path.splitext(f.filename)[1].lower() or ".mp3"
    if akhiran not in EKSTENSI_AUDIO:
        abort(400, description="Format tidak didukung. Gunakan MP3/WAV/FLAC/OGG.")
    ambang, jendela = parameter_uji(request.form)
    fd, tmp = tempfile.mkstemp(suffix=akhiran)
    os.close(fd)
    try:
        f.save(tmp)
        try:
            sinyal = fa.muat_audio(tmp)
        except Exception as e:
            abort(400, description=f"File tidak bisa dibaca: {e}")
    finally:
        os.remove(tmp)
    return jsonify(bersihkan(analisis(sinyal, f.filename, ambang, jendela)))


@app.post("/api/uji-contoh/<nama>")
def uji_contoh(nama):
    if nama not in daftar_contoh():
        abort(404, description="File contoh tidak ditemukan")
    ambang, jendela = parameter_uji(request.get_json(silent=True) or {})
    sinyal = fa.muat_audio(os.path.join(PATH_CONTOH, nama))
    return jsonify(bersihkan(analisis(sinyal, nama, ambang, jendela)))


@app.get("/contoh/<nama>")
def unduh_contoh(nama):
    if nama not in daftar_contoh():
        abort(404)
    return send_file(os.path.join(PATH_CONTOH, nama), max_age=3600)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def galat(e):
    if request.path.startswith("/api/"):
        pesan = "File terlalu besar (maks 50 MB)" if e.code == 413 else e.description
        return jsonify({"error": pesan}), e.code
    return e


# ----------------------------------------------------------------------
def muat(path_model=PATH_MODEL):
    global PAKET, PENCATAT
    if not os.path.exists(path_model):
        raise SystemExit(f"[ERROR] Model tidak ditemukan: {path_model}\n"
                         "Jalankan notebook sampai Bab 9 (Deployment) terlebih dahulu.")
    PAKET = muat_model(path_model)
    PENCATAT = Pencatat(PATH_LOG)
    print(f"Model dimuat: {PAKET['nama_metode']} | kelas {PAKET.get('kelas')} | ambang {ambang_bawaan()}")


if __name__ != "__main__":
    # Dijalankan lewat gunicorn (hosting online): muat model saat modul diimpor.
    # Pakai SATU worker (--workers 1 --threads 8) karena sesi mikrofon disimpan di memori.
    muat(os.environ.get("MODEL_PATH", PATH_MODEL))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web deteksi keramaian lab")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--model", default=PATH_MODEL)
    ap.add_argument("--dev", action="store_true",
                    help="mode pengembangan: server restart otomatis saat file .py/.html diubah")
    args = ap.parse_args()
    muat(args.model)
    if args.dev:
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    print(f"Buka di browser: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=args.dev)
