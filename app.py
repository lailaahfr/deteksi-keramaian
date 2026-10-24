"""
app.py - Sistem web deteksi keramaian lab
=========================================
Server Flask yang memuat model terbaik hasil notebook. Halaman web merekam
mikrofon lab lewat browser, mengirim audio ke server tiap 1 detik, dan
server menilai jendela 5 detik terakhir. Jika lab ramai, halaman memutar
teguran "sssttt... jangan ramai".

Menjalankan:
    python app.py                    # buka http://localhost:5000 di PC lab
    python app.py --port 8000
    python app.py --host 0.0.0.0     # bisa diakses PC lain (lihat README soal HTTPS)

Alur audio (sama dengan training):
    mic browser (sample rate asli, PCM 16-bit) -> buffer 5 detik per sesi
    -> librosa.resample ke 16 kHz -> fitur_audio.ekstrak_fitur (32 fitur)
    -> model.predict_proba -> aturan 3 dari 4 + cooldown -> teguran
"""

import argparse
import collections
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime

import librosa
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

import fitur_audio as fa
from deteksi_lab import Detektor, Pencatat, muat_model

LOKASI = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(LOKASI, "model", "model_terbaik.joblib")
PATH_SUARA = os.path.join(LOKASI, "suara", "peringatan.wav")
PATH_LOG = os.path.join(LOKASI, "hasil", "log_deteksi.csv")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # batas upload 50 MB

PAKET = None          # diisi di muat()
PENCATAT = None
KUNCI_LOG = threading.Lock()
RIWAYAT_TEGURAN = collections.deque(maxlen=200)   # untuk ditampilkan di web
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
        return obj.item()
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


class Sesi:
    """Satu tab browser yang sedang mendengarkan mikrofon."""

    def __init__(self, pengaturan):
        self.kunci = threading.Lock()
        self.buffer = np.zeros(0, dtype=np.float32)   # sample rate asli mic
        self.sr = None
        self.terakhir_aktif = time.time()
        self.detektor = Detektor(PAKET, PAKET.get("ambang_proba", 0.5), 3, 4, 30.0)
        self.atur(pengaturan)

    def atur(self, p):
        d = self.detektor
        d.ambang = angka(p.get("ambang"), d.ambang, float, 0.05, 0.99)
        d.cooldown = angka(p.get("cooldown"), d.cooldown, float, 0, 3600)
        riwayat = angka(p.get("riwayat"), d.riwayat.maxlen, int, 1, 10)
        d.konfirmasi = angka(p.get("konfirmasi"), d.konfirmasi, int, 1, riwayat)
        if riwayat != d.riwayat.maxlen:
            d.riwayat = collections.deque(d.riwayat, maxlen=riwayat)

    def pengaturan(self):
        d = self.detektor
        return dict(ambang=d.ambang, cooldown=d.cooldown,
                    konfirmasi=d.konfirmasi, riwayat=d.riwayat.maxlen)


def ambil_sesi(sid):
    with KUNCI_SESI:
        s = SESI.get(sid)
        # bersihkan sesi yang ditinggal > 10 menit
        batas = time.time() - 600
        for k in [k for k, v in SESI.items() if v.terakhir_aktif < batas]:
            SESI.pop(k, None)
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
    m = PAKET.get("metrik_uji", {})
    return jsonify(bersihkan({
        "metode": PAKET["nama_metode"],
        "parameter": PAKET.get("parameter"),
        "ambang_bawaan": PAKET.get("ambang_proba", 0.5),
        "dibuat": PAKET.get("dibuat"),
        "metrik_uji": m,
        "perbandingan": PAKET.get("perbandingan", []),
        "konfigurasi": {k: v for k, v in PAKET["konfigurasi_fitur"].items() if k != "nama_fitur"},
        "jumlah_fitur": len(PAKET["konfigurasi_fitur"]["nama_fitur"]),
        "ada_suara": os.path.exists(PATH_SUARA),
    }))


@app.get("/suara/peringatan.wav")
def suara():
    if not os.path.exists(PATH_SUARA):
        abort(404)
    return send_file(PATH_SUARA, mimetype="audio/wav", max_age=0)


@app.get("/api/riwayat")
def riwayat():
    return jsonify(list(RIWAYAT_TEGURAN)[:50])


# ----------------------------------------------------------------------
# Deteksi real-time dari mikrofon browser
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
    """Body: PCM int16 little-endian mono. Query: sr = sample rate mic."""
    s = ambil_sesi(sid)
    sr = angka(request.args.get("sr"), 0, int)
    if not 8000 <= sr <= 192000:
        abort(400, description="Parameter sr tidak valid")
    blok = np.frombuffer(request.get_data(), dtype="<i2").astype(np.float32) / 32768.0

    with s.kunci:
        if s.sr != sr:                      # mic berganti -> mulai ulang buffer
            s.sr, s.buffer = sr, np.zeros(0, dtype=np.float32)
        n_jendela = int(fa.DURASI_CHUNK * sr)
        s.buffer = np.concatenate([s.buffer, blok])[-n_jendela:]
        terkumpul = len(s.buffer) / sr
        if len(s.buffer) < n_jendela:
            return jsonify({"siap": False, "terkumpul": round(terkumpul, 1),
                            "dibutuhkan": fa.DURASI_CHUNK})
        jendela = s.buffer if sr == fa.SR else librosa.resample(s.buffer, orig_sr=sr, target_sr=fa.SR)
        t0 = time.perf_counter()
        prob, status, tegur = s.detektor.nilai(jendela, time.time())
        waktu_ms = (time.perf_counter() - t0) * 1000
        jumlah_berisik = int(sum(s.detektor.riwayat))
        panjang_riwayat = len(s.detektor.riwayat)

    catat(prob, status, tegur)
    level_db = float(20 * np.log10(np.sqrt(np.mean(jendela[-fa.SR:] ** 2)) + 1e-10))
    return jsonify({"siap": True, "prob": round(prob, 4), "status": status, "tegur": bool(tegur),
                    "ambang": s.detektor.ambang, "level_db": round(level_db, 1),
                    "riwayat_berisik": jumlah_berisik, "riwayat_isi": panjang_riwayat,
                    "waktu_ms": round(waktu_ms, 1)})


# ----------------------------------------------------------------------
# Uji file audio (MP3/WAV) dari halaman web
# ----------------------------------------------------------------------
@app.post("/api/uji-file")
def uji_file():
    f = request.files.get("file")
    if f is None or not f.filename:
        abort(400, description="Tidak ada file yang diunggah")
    akhiran = os.path.splitext(f.filename)[1].lower() or ".mp3"
    if akhiran not in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        abort(400, description="Format tidak didukung. Gunakan MP3/WAV/FLAC/OGG.")
    ambang = angka(request.form.get("ambang"), PAKET.get("ambang_proba", 0.5), float, 0.05, 0.99)

    fd, tmp = tempfile.mkstemp(suffix=akhiran)
    os.close(fd)
    try:
        f.save(tmp)
        try:
            y = fa.muat_audio(tmp)
        except Exception as e:
            abort(400, description=f"File tidak bisa dibaca: {e}")
    finally:
        os.remove(tmp)

    durasi = len(y) / fa.SR
    model = PAKET["model"]

    # (a) per potongan 5 detik - sama seperti data uji di notebook
    chunks = fa.potong_chunk(y)
    potongan = []
    if chunks:
        X = np.vstack([fa.ekstrak_fitur(c) for c in chunks])
        for i, p in enumerate(model.predict_proba(X)[:, 1]):
            potongan.append({"mulai": i * fa.DURASI_CHUNK,
                             "selesai": min((i + 1) * fa.DURASI_CHUNK, durasi),
                             "prob": round(float(p), 4),
                             "status": "BERISIK" if p >= ambang else "tenang"})

    # (b) simulasi sistem: jendela 5 detik digeser tiap `langkah` detik + aturan 3 dari 4
    n = int(fa.DURASI_CHUNK * fa.SR)
    y_sim = np.pad(y, (0, max(0, n - len(y))))
    langkah = max(1.0, durasi / 300)                  # maksimal +-300 titik
    det = Detektor(PAKET, ambang, 3, 4, 30.0)
    simulasi = []
    for awal in range(0, len(y_sim) - n + 1, int(langkah * fa.SR)):
        t = (awal + n) / fa.SR
        p, status, tegur = det.nilai(y_sim[awal:awal + n], t)
        simulasi.append({"t": round(t, 1), "prob": round(p, 4), "tegur": bool(tegur)})

    persen = 100 * np.mean([c["status"] == "BERISIK" for c in potongan]) if potongan else 0.0
    return jsonify(bersihkan({
        "nama": f.filename, "durasi": round(durasi, 1), "ambang": ambang,
        "potongan": potongan, "simulasi": simulasi,
        "persen_berisik": round(float(persen), 1),
        "keputusan": "BERISIK" if persen >= 50 else "TIDAK BERISIK",
        "jumlah_teguran": int(sum(s["tegur"] for s in simulasi)),
    }))


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
                         "Jalankan notebook sampai tahap Deployment terlebih dahulu.")
    PAKET = muat_model(path_model)
    PENCATAT = Pencatat(PATH_LOG)
    print(f"Model dimuat: {PAKET['nama_metode']} | ambang {PAKET.get('ambang_proba', 0.5)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web deteksi keramaian lab")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model", default=PATH_MODEL)
    args = ap.parse_args()
    muat(args.model)
    print(f"Buka di browser: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
