#!/usr/bin/env python3
import os
import sys
import json
from typing import Tuple

import numpy as np

TRAIN_WAV = "road_sound_test1.wav"
VAL_WAV = "road_sound_test2.wav"

def _log(msg: str) -> None:
    print(f"[preprocess] {msg}")

def try_import_audio_lib():
    try:
        import librosa  # type: ignore
        return "librosa"
    except Exception:
        try:
            from scipy.io import wavfile  # type: ignore
            return "scipy"
        except Exception:
            return None

def load_wav_as_mono(path: str) -> Tuple[np.ndarray, int]:
    lib = try_import_audio_lib()
    if lib is None:
        raise RuntimeError("librosa/scipy 둘 다 없음. `pip install librosa` 또는 `pip install scipy` 해라.")

    if lib == "librosa":
        import librosa  # type: ignore
        _log(f"librosa 로딩 사용: {path}")
        wav, sr = librosa.load(path, sr=None, mono=True)
        return wav.astype(np.float32), int(sr)
    else:
        from scipy.io import wavfile  # type: ignore
        _log(f"scipy.io.wavfile 로딩 사용: {path}")
        sr, wav = wavfile.read(path)
        wav = wav.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return wav, int(sr)

def compute_rms_per_second(wav: np.ndarray, sr: int) -> np.ndarray:
    samples_per_sec = sr
    total_samples = len(wav)
    n_full_secs = total_samples // samples_per_sec
    if n_full_secs == 0:
        raise ValueError("오디오 길이가 1초보다 짧음.")
    wav = wav[: n_full_secs * samples_per_sec]
    frames = wav.reshape(n_full_secs, samples_per_sec)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return rms.astype(np.float32)

def discretize_energy(rms_values: np.ndarray) -> np.ndarray:
    if len(rms_values) < 5:
        # 너무 짧으면 그냥 값 크기 기준으로 정렬해 대충 나눔
        order = np.argsort(rms_values)
        bins = np.zeros_like(order, dtype=np.int64)
        # 제일 큰 1개는 4, 그다음 3, 나머지 0~2 정도로
        unique = len(order)
        if unique >= 1:
            bins[order[-1]] = 4
        if unique >= 2:
            bins[order[-2]] = 3
        if unique >= 3:
            bins[order[-3]] = 2
        return bins

    q = np.quantile(rms_values, [0.2, 0.4, 0.6, 0.8])
    bins = np.zeros_like(rms_values, dtype=np.int64)

    for i, v in enumerate(rms_values):
        if v < q[0]:
            bins[i] = 0
        elif v < q[1]:
            bins[i] = 1
        elif v < q[2]:
            bins[i] = 2
        elif v < q[3]:
            bins[i] = 3
        else:
            bins[i] = 4
    return bins

def process_one(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"wav 파일 없음: {path}")
    _log(f"로드: {path}")
    wav, sr = load_wav_as_mono(path)
    _log(f"샘플 수={len(wav)}, 샘플레이트={sr}")
    rms = compute_rms_per_second(wav, sr)
    _log(f"1초 RMS 길이={len(rms)}")
    bins = discretize_energy(rms)
    _log(f"에너지 bin 분포: " +
         ", ".join(f"{b}: {(bins==b).sum()}" for b in range(5)))
    return bins

def main():
    try:
        train_bins = process_one(TRAIN_WAV)
        np.save("train_bins.npy", train_bins)
        _log(f"train_bins.npy 저장 완료 (len={len(train_bins)})")

        val_bins = process_one(VAL_WAV)
        np.save("val_bins.npy", val_bins)
        _log(f"val_bins.npy 저장 완료 (len={len(val_bins)})")

        meta = {
            "train_len": int(len(train_bins)),
            "val_len": int(len(val_bins)),
        }
        with open("preprocess_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _log("preprocess_meta.json 저장 완료")
    except Exception as e:
        _log(f"에러 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

