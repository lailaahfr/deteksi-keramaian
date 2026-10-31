"""
fitur_audio.py
==============
Pipeline pra-pemrosesan dan ekstraksi fitur yang SAMA PERSIS dengan notebook
`klasifikasi-audio.ipynb` (Bab 3-5). Dipakai oleh sistem web agar fitur saat
deteksi identik dengan fitur saat training.

    librosa.load(sr=16000, res_type='polyphase')            (3.1 resample)
    -> potong chunk 5 detik, sisa < 5 detik dibuang           (3.2 chunking)
    -> frame RMS < -80 dBFS = hening; chunk dibuang jika
       frame aktif < 50%                                      (3.3 buang hening)
    -> 16 deret per frame aktif: RMS, centroid, ZCR, MFCC1-13 (4.5)
    -> gabung frame aktif semua chunk, ambil mean lalu std    (5.2)
    => 32 fitur, urutan: 16 *_mean lalu 16 *_std

Jangan ubah angka di bawah tanpa melatih ulang model lewat notebook.
"""

import librosa
import numpy as np

SR = 16000
RES_TYPE = "polyphase"
DURASI_CHUNK = 5
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13
AMBANG_DB = -80
MIN_FRAME_AKTIF = 0.5

NAMA_DERET = ["rms", "centroid", "zcr"] + [f"mfcc{i}" for i in range(1, 14)]
FITUR = [f"{n}_mean" for n in NAMA_DERET] + [f"{n}_std" for n in NAMA_DERET]


def muat_audio(path):
    """Baca file audio (mp3/wav) -> mono 16 kHz, resample polifase (sama dengan notebook)."""
    sinyal, _ = librosa.load(path, sr=SR, res_type=RES_TYPE)
    return sinyal


def resample(sinyal, sr_asal):
    """Resample sinyal mikrofon ke 16 kHz dengan metode yang sama dengan librosa.load di notebook."""
    if sr_asal == SR:
        return np.asarray(sinyal, dtype=np.float32)
    return librosa.resample(np.asarray(sinyal, dtype=np.float32), orig_sr=sr_asal,
                            target_sr=SR, res_type=RES_TYPE)


def potong_chunk(sinyal, sr=SR, durasi=DURASI_CHUNK):
    """Bagi sinyal menjadi potongan berdurasi tetap; sisa di akhir yang kurang dari durasi dibuang."""
    panjang = int(durasi * sr)
    return [sinyal[i:i + panjang] for i in range(0, len(sinyal) - panjang + 1, panjang)]


def frame_aktif(chunk):
    """Mask frame yang energinya di atas ambang hening."""
    rms = librosa.feature.rms(y=chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    return 20 * np.log10(rms + 1e-10) > AMBANG_DB


def praproses_sinyal(sinyal, sr=SR):
    """Sinyal 16 kHz -> daftar (chunk, mask_aktif) yang layak dipakai."""
    hasil = []
    for chunk in potong_chunk(sinyal, sr):
        aktif = frame_aktif(chunk)
        if aktif.mean() >= MIN_FRAME_AKTIF:
            hasil.append((chunk, aktif))
    return hasil


def fitur_per_frame(chunk, aktif, sr=SR):
    """16 deret nilai per frame (hanya frame aktif)."""
    rms = librosa.feature.rms(y=chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    semua = np.vstack([rms, centroid, zcr, mfcc])
    return semua[:, aktif]


def ekstrak_sinyal(sinyal, sr=SR):
    """Sinyal 16 kHz -> dict 32 fitur (+ jumlah_chunk), atau None jika seluruhnya hening.

    Sama dengan `ekstrak_file` di notebook, hanya saja menerima sinyal yang sudah dimuat.
    """
    hasil = praproses_sinyal(sinyal, sr)
    if not hasil:
        return None
    deret = np.hstack([fitur_per_frame(chunk, aktif, sr) for chunk, aktif in hasil])
    fitur = {}
    for nama, nilai in zip(NAMA_DERET, deret):
        fitur[f"{nama}_mean"] = nilai.mean()
    for nama, nilai in zip(NAMA_DERET, deret):
        fitur[f"{nama}_std"] = nilai.std()
    fitur["jumlah_chunk"] = len(hasil)
    return fitur


def ekstrak_file(path):
    """Satu file audio -> dict 32 fitur (identik dengan `ekstrak_file` di notebook)."""
    return ekstrak_sinyal(muat_audio(path))


def konfigurasi():
    """Konfigurasi pipeline, disimpan bersama model untuk dicek kecocokannya."""
    return dict(sr=SR, res_type=RES_TYPE, durasi_chunk=DURASI_CHUNK, frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH, n_mfcc=N_MFCC, ambang_hening_db=AMBANG_DB,
                min_frame_aktif=MIN_FRAME_AKTIF, urutan_fitur=list(FITUR))
