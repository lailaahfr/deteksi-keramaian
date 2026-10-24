"""
fitur_audio.py
==============
Modul bersama untuk NOTEBOOK (training) dan SISTEM (deteksi real-time).

Kenapa dipisah jadi modul sendiri?
Fitur yang dihitung saat training HARUS identik dengan fitur saat deteksi di lab.
Kalau kodenya disalin dua kali, cepat atau lambat akan beda (misal hop_length
berubah di satu tempat saja) dan model jadi salah tebak tanpa ketahuan.

Pipeline (sesuai modul):
    Rekaman audio -> Pra-pemrosesan (resample 16 kHz, chunk 5-10 detik, buang hening)
    -> Ekstraksi fitur per frame (RMS, Spectral Centroid, ZCR, 13 MFCC)
    -> Agregasi mean & std per chunk  => vektor 32 fitur
"""

import numpy as np
import librosa

# ------------------------------------------------------------------
# Konfigurasi pipeline (disimpan juga di dalam file model .joblib)
# ------------------------------------------------------------------
SR = 16000            # sample rate tujuan (Hz) - tips modul: resample ke 16 kHz
DURASI_CHUNK = 5.0    # panjang potongan (detik) - tips modul: 5-10 detik
DURASI_MIN = 2.0      # sisa potongan di ujung rekaman dipakai jika >= 2 detik
FRAME_LENGTH = 2048   # panjang frame (sampel) untuk semua fitur
HOP_LENGTH = 512      # pergeseran frame (sampel)
N_MFCC = 13
AMBANG_HENING_DB = -60.0  # frame dengan RMS < -60 dBFS dianggap hening (dibuang)
MIN_FRAME_AKTIF = 10      # jika frame aktif < 10, pakai semua frame (chunk memang sunyi)

NAMA_FITUR = (
    ["rms_mean", "rms_std",
     "centroid_mean", "centroid_std",
     "zcr_mean", "zcr_std"]
    + [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
)  # total 2 + 2 + 2 + 13 + 13 = 32 fitur


def muat_audio(path, sr=SR):
    """Membaca file audio (wav/mp3/flac/ogg), mono, di-resample ke `sr`."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def potong_chunk(y, sr=SR, durasi=DURASI_CHUNK, durasi_min=DURASI_MIN):
    """Memotong sinyal menjadi potongan berdurasi tetap (tanpa overlap).

    Rekaman yang lebih pendek dari `durasi` tetap dipakai utuh selama
    panjangnya >= `durasi_min`.
    """
    n = int(durasi * sr)
    n_min = int(durasi_min * sr)
    if len(y) <= n:
        return [y] if len(y) >= n_min else []
    potongan = []
    for awal in range(0, len(y), n):
        seg = y[awal:awal + n]
        if len(seg) >= n_min:
            potongan.append(seg)
    return potongan


def ekstrak_fitur(y, sr=SR):
    """Ekstraksi fitur per frame lalu agregasi mean & std -> vektor 32 fitur.

    Pembuangan hening (tips 1 modul): fitur hanya dihitung dari frame yang
    energinya di atas AMBANG_HENING_DB. Ambangnya sengaja rendah (-60 dBFS)
    karena untuk kasus ini "sepi" adalah informasi penting bagi kelas
    non-berisik; yang dibuang hanya bagian yang benar-benar hening
    (jeda/dropout rekaman), bukan suara latar AC/kipas.
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < FRAME_LENGTH:
        y = np.pad(y, (0, FRAME_LENGTH - len(y)))

    kw = dict(hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, **kw)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, **kw)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, **kw)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, **kw)

    # samakan jumlah frame (jaga-jaga beda 1 frame antar fungsi)
    n = min(len(rms), len(cent), len(zcr), mfcc.shape[1])
    rms, cent, zcr, mfcc = rms[:n], cent[:n], zcr[:n], mfcc[:, :n]

    # buang frame hening
    rms_db = 20 * np.log10(rms + 1e-10)
    aktif = rms_db > AMBANG_HENING_DB
    if aktif.sum() < MIN_FRAME_AKTIF:
        aktif = np.ones(n, dtype=bool)
    rms, cent, zcr, mfcc = rms[aktif], cent[aktif], zcr[aktif], mfcc[:, aktif]

    fitur = np.hstack([
        np.mean(rms), np.std(rms),
        np.mean(cent), np.std(cent),
        np.mean(zcr), np.std(zcr),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
    ]).astype(np.float32)
    return np.nan_to_num(fitur)


def fitur_dari_file(path, sr=SR, durasi=DURASI_CHUNK):
    """Satu file audio -> matriks fitur (jumlah_chunk x 32)."""
    y = muat_audio(path, sr=sr)
    chunks = potong_chunk(y, sr=sr, durasi=durasi)
    if not chunks:
        return np.empty((0, len(NAMA_FITUR)), dtype=np.float32)
    return np.vstack([ekstrak_fitur(c, sr=sr) for c in chunks])


def konfigurasi():
    """Konfigurasi pipeline yang disimpan bersama model."""
    return dict(sr=SR, durasi_chunk=DURASI_CHUNK, frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH, n_mfcc=N_MFCC,
                ambang_hening_db=AMBANG_HENING_DB, nama_fitur=list(NAMA_FITUR))
