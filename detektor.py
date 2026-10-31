"""
detektor.py - memuat model dari notebook dan memutuskan kapan menegur
=====================================================================
Dipakai oleh app.py (sistem web).
"""

import collections
import csv
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import fitur_audio as fa


def muat_model(path):
    """Muat paket model hasil Bab 9 notebook dan pastikan pipeline-nya sama dengan fitur_audio.py."""
    paket = joblib.load(path)
    simpan = paket.get("konfigurasi_fitur", {})
    sekarang = fa.konfigurasi()
    beda = [k for k in sekarang if simpan.get(k) != sekarang[k]]
    if beda:
        raise SystemExit(f"[ERROR] Konfigurasi fitur model berbeda dengan fitur_audio.py: {beda}. "
                         "Samakan fitur_audio.py dengan notebook lalu simpan ulang model.")
    kelas = list(paket["model"].classes_)
    if paket["kelas_berisik"] not in kelas:
        raise SystemExit(f"[ERROR] Kelas '{paket['kelas_berisik']}' tidak ada di model (kelas: {kelas}).")
    paket["_idx_berisik"] = kelas.index(paket["kelas_berisik"])
    return paket


def prob_berisik(paket, fitur):
    """dict 32 fitur -> probabilitas kelas berisik (urutan kolom sama dengan FITUR di notebook)."""
    X = pd.DataFrame([[fitur[k] for k in paket["fitur"]]], columns=paket["fitur"])
    return float(paket["model"].predict_proba(X)[0, paket["_idx_berisik"]])


def nilai_sinyal(paket, sinyal):
    """Sinyal 16 kHz -> (prob_berisik, status). status: 'BERISIK' / 'tenang' / 'hening'.

    'hening' = semua chunk dibuang oleh aturan buang hening (dianggap tidak berisik).
    """
    fitur = fa.ekstrak_sinyal(sinyal)
    if fitur is None:
        return 0.0, "hening"
    return prob_berisik(paket, fitur), None


class Detektor:
    """Aturan teguran: tegur jika `konfirmasi` dari `riwayat` penilaian terakhir berisik,
    dengan jeda minimal `cooldown` detik antar teguran."""

    def __init__(self, paket, ambang=0.5, konfirmasi=3, riwayat=4, cooldown=30.0):
        self.paket = paket
        self.ambang = ambang
        self.konfirmasi = konfirmasi
        self.riwayat = collections.deque(maxlen=riwayat)
        self.cooldown = cooldown
        self.terakhir_tegur = -1e9

    def nilai(self, sinyal, sekarang):
        """Mengembalikan (prob_berisik, status, harus_tegur)."""
        prob, status = nilai_sinyal(self.paket, sinyal)
        berisik = status != "hening" and prob >= self.ambang
        if status is None:
            status = "BERISIK" if berisik else "tenang"
        self.riwayat.append(berisik)
        tegur = sum(self.riwayat) >= self.konfirmasi and (sekarang - self.terakhir_tegur) >= self.cooldown
        if tegur:
            self.terakhir_tegur = sekarang
            self.riwayat.clear()
        return prob, status, tegur


class Pencatat:
    """Menulis riwayat deteksi ke CSV."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        baru = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if baru:
            self.w.writerow(["waktu", "prob_berisik", "status", "teguran"])

    def tulis(self, prob, status, teguran):
        self.w.writerow([datetime.now().isoformat(timespec="seconds"), f"{prob:.3f}", status, int(teguran)])
        self.f.flush()
