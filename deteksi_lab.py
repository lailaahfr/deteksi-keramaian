"""
deteksi_lab.py - Sistem deteksi keramaian lab secara real-time
===============================================================
Mendengarkan mikrofon, mengklasifikasi suasana lab dengan model terbaik hasil
notebook, lalu memutar teguran "sssttt... jangan ramai" bila lab berisik.

Contoh pemakaian:
    python deteksi_lab.py                         # pakai mikrofon default
    python deteksi_lab.py --list-devices          # lihat daftar mikrofon/speaker
    python deteksi_lab.py --input 2 --output 4    # pilih perangkat tertentu
    python deteksi_lab.py --file rekaman.wav      # simulasi dari file (tanpa mikrofon)
    python deteksi_lab.py --ambang 0.8 --cooldown 60

Tekan Ctrl+C untuk berhenti.
"""

import argparse
import collections
import csv
import os
import queue
import sys
import time
from datetime import datetime

import joblib
import numpy as np

import fitur_audio as fa

LOKASI = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------
def muat_model(path):
    paket = joblib.load(path)
    if paket["konfigurasi_fitur"] != fa.konfigurasi():
        sys.exit("[ERROR] Konfigurasi fitur di model berbeda dengan fitur_audio.py. "
                 "Latih ulang model lewat notebook.")
    return paket


def suara_ssst_sintetis(sr=fa.SR, durasi=1.6):
    """Cadangan bila suara/peringatan.wav belum ada: desis 'sssttt' dari noise tersaring."""
    rng = np.random.default_rng(0)
    n = int(sr * durasi)
    noise = rng.normal(0, 1, n)
    spek = np.fft.rfft(noise)
    frek = np.fft.rfftfreq(n, 1 / sr)
    spek[(frek < 3500) | (frek > 7500)] = 0          # pita desis 's'
    desis = np.fft.irfft(spek, n)
    t = np.linspace(0, 1, n)
    selubung = np.minimum(1, t / 0.08) * np.minimum(1, (1 - t) / 0.25)
    desis = desis * selubung
    return (0.6 * desis / (np.abs(desis).max() + 1e-9)).astype(np.float32), sr


def muat_suara_peringatan(path):
    if path and os.path.exists(path):
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), sr
    print(f"[INFO] {path} tidak ditemukan -> memakai desis 'sssttt' sintetis. "
          f"Jalankan buat_suara_peringatan.py untuk suara lengkap.")
    return suara_ssst_sintetis()


class Pencatat:
    """Menulis riwayat deteksi ke CSV."""
    def __init__(self, path):
        self.path = path
        baru = not os.path.exists(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if baru:
            self.w.writerow(["waktu", "prob_berisik", "status", "teguran"])

    def tulis(self, prob, status, teguran):
        self.w.writerow([datetime.now().isoformat(timespec="seconds"), f"{prob:.3f}", status, int(teguran)])
        self.f.flush()

    def tutup(self):
        self.f.close()


# ----------------------------------------------------------------------
# Inti keputusan (dipakai mode mikrofon dan mode file)
# ----------------------------------------------------------------------
class Detektor:
    def __init__(self, paket, ambang, konfirmasi, riwayat, cooldown):
        self.model = paket["model"]
        self.ambang = ambang
        self.konfirmasi = konfirmasi            # jumlah penilaian "berisik" yang dibutuhkan ...
        self.riwayat = collections.deque(maxlen=riwayat)   # ... dari sekian penilaian terakhir
        self.cooldown = cooldown
        self.terakhir_tegur = -1e9

    def nilai(self, jendela, sekarang):
        """Mengembalikan (prob_berisik, status, harus_tegur)."""
        x = fa.ekstrak_fitur(jendela, sr=fa.SR).reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0, 1])
        berisik = prob >= self.ambang
        self.riwayat.append(berisik)
        cukup = sum(self.riwayat) >= self.konfirmasi
        siap = (sekarang - self.terakhir_tegur) >= self.cooldown
        tegur = cukup and siap
        if tegur:
            self.terakhir_tegur = sekarang
            self.riwayat.clear()
        status = "BERISIK" if berisik else "tenang"
        return prob, status, tegur


def tampilkan(prob, status, tegur, ambang, info=""):
    panjang = 30
    isi = int(round(prob * panjang))
    batang = "#" * isi + "-" * (panjang - isi)
    tanda = "  >>> sssttt... jangan ramai! <<<" if tegur else ""
    print(f"{datetime.now():%H:%M:%S} [{batang}] {prob:5.2f} (ambang {ambang:.2f}) "
          f"{status:8s}{info}{tanda}", flush=True)


# ----------------------------------------------------------------------
# Mode 1: mikrofon real-time
# ----------------------------------------------------------------------
def jalankan_mikrofon(args, paket, detektor, suara, pencatat):
    import sounddevice as sd
    import librosa

    info_in = sd.query_devices(args.input, "input")
    sr_mic = fa.SR
    try:
        sd.check_input_settings(device=args.input, samplerate=fa.SR, channels=1)
    except Exception:
        sr_mic = int(info_in["default_samplerate"])   # mic tidak mendukung 16 kHz -> resample
    n_jendela = int(fa.DURASI_CHUNK * fa.SR)

    antrean = queue.Queue()
    def callback(indata, frames, waktu, status):
        antrean.put(indata[:, 0].copy())

    buffer = np.zeros(0, dtype=np.float32)
    y_suara, sr_suara = suara
    print(f"Mikrofon : {info_in['name']} ({sr_mic} Hz)")
    print(f"Model    : {paket['nama_metode']} | ambang {detektor.ambang} | "
          f"tegur jika {detektor.konfirmasi} dari {detektor.riwayat.maxlen} penilaian berisik | "
          f"cooldown {detektor.cooldown} s")
    print("Mengumpulkan 5 detik audio pertama... (Ctrl+C untuk berhenti)\n")

    with sd.InputStream(device=args.input, channels=1, samplerate=sr_mic,
                        blocksize=int(sr_mic * 0.1), dtype="float32", callback=callback):
        berikut = time.time() + args.interval
        while True:
            # ambil semua blok audio baru
            while not antrean.empty():
                blok = antrean.get()
                if sr_mic != fa.SR:
                    blok = librosa.resample(blok, orig_sr=sr_mic, target_sr=fa.SR)
                buffer = np.concatenate([buffer, blok])[-n_jendela:]
            if time.time() < berikut:
                time.sleep(0.05)
                continue
            berikut += args.interval
            if len(buffer) < n_jendela:
                continue

            prob, status, tegur = detektor.nilai(buffer, time.time())
            tampilkan(prob, status, tegur, detektor.ambang)
            pencatat.tulis(prob, status, tegur)

            if tegur:
                sd.play(y_suara * args.volume, sr_suara, device=args.output)
                sd.wait()
                # buang audio yang terekam selama teguran diputar (agar tidak mendeteksi diri sendiri)
                time.sleep(0.3)
                while not antrean.empty():
                    antrean.get()
                buffer = np.zeros(0, dtype=np.float32)
                berikut = time.time() + args.interval


# ----------------------------------------------------------------------
# Mode 2: simulasi dari file audio (untuk uji tanpa mikrofon)
# ----------------------------------------------------------------------
def jalankan_file(args, paket, detektor, suara, pencatat):
    y = fa.muat_audio(args.file)
    n_jendela = int(fa.DURASI_CHUNK * fa.SR)
    langkah = int(args.interval * fa.SR)
    if len(y) < n_jendela:
        y = np.pad(y, (0, n_jendela - len(y)))
    print(f"Simulasi file: {args.file} ({len(y) / fa.SR:.1f} detik) | model {paket['nama_metode']}\n")
    jumlah_tegur = 0
    for awal in range(0, len(y) - n_jendela + 1, langkah):
        detik = (awal + n_jendela) / fa.SR
        prob, status, tegur = detektor.nilai(y[awal:awal + n_jendela], detik)
        tampilkan(prob, status, tegur, detektor.ambang, info=f" t={detik:6.1f}s")
        pencatat.tulis(prob, status, tegur)
        jumlah_tegur += tegur
        if tegur and args.putar:
            import sounddevice as sd
            sd.play(suara[0] * args.volume, suara[1], device=args.output)
            sd.wait()
    print(f"\nSelesai. Jumlah teguran: {jumlah_tegur}")
    return jumlah_tegur


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Deteksi keramaian lab + teguran suara")
    p.add_argument("--model", default=os.path.join(LOKASI, "model", "model_terbaik.joblib"))
    p.add_argument("--suara", default=os.path.join(LOKASI, "suara", "peringatan.wav"))
    p.add_argument("--log", default=os.path.join(LOKASI, "hasil", "log_deteksi.csv"))
    p.add_argument("--ambang", type=float, default=None,
                   help="ambang probabilitas berisik (default: dari model)")
    p.add_argument("--interval", type=float, default=1.0, help="jeda antar penilaian (detik)")
    p.add_argument("--konfirmasi", type=int, default=3, help="jumlah penilaian berisik yang dibutuhkan")
    p.add_argument("--riwayat", type=int, default=4, help="dari sekian penilaian terakhir")
    p.add_argument("--cooldown", type=float, default=30.0, help="jeda minimum antar teguran (detik)")
    p.add_argument("--volume", type=float, default=1.0, help="pengali volume teguran")
    p.add_argument("--input", type=int, default=None, help="nomor perangkat mikrofon")
    p.add_argument("--output", type=int, default=None, help="nomor perangkat speaker")
    p.add_argument("--file", default=None, help="simulasi memakai file audio, bukan mikrofon")
    p.add_argument("--putar", action="store_true", help="mode --file: tetap putar suara teguran")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model tidak ditemukan: {args.model}\n"
                 f"Jalankan notebook sampai tahap Deployment terlebih dahulu.")
    paket = muat_model(args.model)
    ambang = args.ambang if args.ambang is not None else paket.get("ambang_proba", 0.5)
    detektor = Detektor(paket, ambang, args.konfirmasi, args.riwayat, args.cooldown)
    suara = muat_suara_peringatan(args.suara)
    pencatat = Pencatat(args.log)

    try:
        if args.file:
            jalankan_file(args, paket, detektor, suara, pencatat)
        else:
            jalankan_mikrofon(args, paket, detektor, suara, pencatat)
    except KeyboardInterrupt:
        print("\nDihentikan.")
    finally:
        pencatat.tutup()
        print(f"Log tersimpan di {args.log}")


if __name__ == "__main__":
    main()
