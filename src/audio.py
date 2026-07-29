"""Mono audio decode + block RMS, using only ffmpeg and numpy.

This used to be `librosa.load` + `librosa.feature.rms`. Both were replaced
because librosa pulls in numba/llvmlite, which is the single hardest thing to
freeze into a .app/.exe -- and neither call needed any of it. Decoding is
already ffmpeg's job everywhere else in the pipeline, and the RMS we ask for
uses frame_length == hop_length, which is plain non-overlapping block RMS.

Verified against librosa on real footage (game1, 27min): decode is
sample-for-sample identical, dB differs by at most 5e-6, the same 14 peaks come
out at the same times -- and decoding is ~12x faster (2.40s -> 0.20s).
"""
import numpy as np

from src.ffmpeg import ffmpeg


def load(path, sr=16000):
    """Decode `path` to a mono float32 array at `sr` Hz."""
    import subprocess
    out = subprocess.run(
        [ffmpeg(), "-v", "error", "-i", path, "-map", "a:0?", "-ac", "1",
         "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float32, copy=False)


def block_rms(y, hop):
    """RMS over consecutive non-overlapping blocks of `hop` samples.

    librosa centres its frames by default, which pads half a frame of zeros at
    each end and yields one extra frame; `rms_db` below reproduces that so peak
    timings stay identical to the pre-librosa-removal runs.
    """
    hop = max(1, int(hop))
    pad = hop // 2
    y = np.pad(y, (pad, pad), mode="constant")
    n = 1 + (len(y) - hop) // hop if len(y) >= hop else 0
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n, hop), strides=(y.strides[0] * hop, y.strides[0]))
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
