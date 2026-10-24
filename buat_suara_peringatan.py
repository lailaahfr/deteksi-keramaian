"""
buat_suara_peringatan.py - membuat suara/peringatan.wav ("sssttt... jangan ramai")
==================================================================================
Pilihan:
    python buat_suara_peringatan.py            # desis "sssttt" + suara Google TTS "jangan ramai" (butuh internet)
    python buat_suara_peringatan.py --rekam 3  # rekam suara sendiri selama 3 detik lewat mikrofon
    python buat_suara_peringatan.py --tes      # putar file yang sudah ada

Paling natural: rekam suara sendiri (atau suara asisten lab) mengucapkan "sssttt... jangan ramai".
Anda juga bisa langsung menaruh file WAV buatan sendiri dengan nama suara/peringatan.wav.
"""

import argparse
import os
import tempfile

import numpy as np
import soundfile as sf

from deteksi_lab import suara_ssst_sintetis

LOKASI = os.path.dirname(os.path.abspath(__file__))
TUJUAN = os.path.join(LOKASI, "suara", "peringatan.wav")
SR = 22050


def dari_tts(teks="jangan ramai"):
    from gtts import gTTS
    import librosa
    tmp = os.path.join(tempfile.gettempdir(), "tts_peringatan.mp3")
    gTTS(teks, lang="id").save(tmp)
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y


def rekam(detik):
    import sounddevice as sd
    print(f"Bersiap... ucapkan 'sssttt... jangan ramai' setelah tanda MULAI ({detik} detik)")
    import time; time.sleep(1.0)
    print("MULAI")
    y = sd.rec(int(detik * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("Selesai merekam.")
    import librosa
    y, _ = librosa.effects.trim(y[:, 0], top_db=35)
    return y


def normalisasi(y, puncak=0.9):
    return (puncak * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rekam", type=float, default=None, help="rekam suara sendiri selama N detik")
    p.add_argument("--teks", default="jangan ramai", help="teks TTS setelah desis 'sssttt'")
    p.add_argument("--tes", action="store_true", help="putar suara/peringatan.wav")
    args = p.parse_args()
    os.makedirs(os.path.dirname(TUJUAN), exist_ok=True)

    if args.tes:
        import sounddevice as sd
        y, sr = sf.read(TUJUAN, dtype="float32")
        sd.play(y, sr); sd.wait()
        return

    if args.rekam:
        y = rekam(args.rekam)
    else:
        import librosa
        desis, sr_desis = suara_ssst_sintetis(durasi=1.4)
        desis = librosa.resample(desis, orig_sr=sr_desis, target_sr=SR)
        jeda = np.zeros(int(0.25 * SR), dtype=np.float32)
        try:
            ucapan = dari_tts(args.teks)
            y = np.concatenate([desis * 0.5, jeda, normalisasi(ucapan)])
            print("Memakai desis sintetis + Google TTS.")
        except Exception as e:
            print(f"[INFO] TTS gagal ({e}). Hanya desis 'sssttt' yang dibuat.\n"
                  f"       Disarankan rekam suara sendiri: python buat_suara_peringatan.py --rekam 3")
            y = desis

    sf.write(TUJUAN, normalisasi(y), SR)
    print(f"Tersimpan: {TUJUAN} ({len(y) / SR:.1f} detik). Coba putar: python buat_suara_peringatan.py --tes")


if __name__ == "__main__":
    main()
