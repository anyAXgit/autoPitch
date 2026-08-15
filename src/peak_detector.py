"""Find the loud moments.

Loudness is measured in `k` -- standard deviations above that recording's own
median -- never in raw dB. Two microphones at opposite corners of a pitch have
different gain, different distance to the crowd and different noise floors, so
their dB are not comparable; their k are.
"""
import numpy as np

from src.audio import block_rms, load


def rms_db(path, window_sec, sr=16000):
    y = load(path, sr)
    hop = max(1, int(window_sec * sr))
    rms = block_rms(y, hop)
    db = 20.0 * np.log10(rms + 1e-8)
    times = np.arange(len(db)) * hop / sr
    return times, db


def k_envelope(path, window_sec, sr=16000):
    """Loudness over time, in standard deviations above this file's median."""
    times, db = rms_db(path, window_sec, sr)
    sd = float(np.std(db))
    if sd <= 0:
        return times, np.zeros_like(db)      # a constant signal has no peaks
    return times, (db - float(np.median(db))) / sd


def _suppress(cand, cfg):
    """One survivor per `min_gap_sec` cluster: the loudest.

    `cand` is [(time, k)], sorted by time.
    """
    kept = []
    for t, v in cand:
        if kept and t - kept[-1][0] < cfg.min_gap_sec:
            # Round before comparing so sub-0.1dB float/codec noise on an
            # otherwise flat burst plateau doesn't let a later frame "win"
            # by a razor-thin margin; ties resolve to the earlier (onset)
            # frame, which is the semantically correct goal moment.
            if round(v, 1) > round(kept[-1][1], 1):
                kept[-1] = (t, v)
        else:
            kept.append((t, v))
    if cfg.max_clips is not None and len(kept) > cfg.max_clips:
        kept = sorted(kept, key=lambda c: c[1], reverse=True)[: cfg.max_clips]
    return sorted(t for t, _ in kept)


def detect_peaks(path, cfg, sr=16000):
    """Peaks from a single recording."""
    times, k = k_envelope(path, cfg.rms_window_sec, sr)
    over = k >= cfg.threshold_k
    return _suppress(list(zip(times[over], k[over])), cfg)


def detect_peaks_multicam(audio_by_cam, offsets, ref, cfg, sr=16000):
    """Peaks from every camera at once, reported on `ref`'s clock.

    A goal at the far end of the pitch is loud on the camera beside it and
    faint on the one across from it. Listening to a single camera therefore
    throws away half the match -- measured on one game here, a cheer at k=4.76
    on cam2 registered 1.53 on cam1 and was never proposed at all. Take the
    loudest camera per frame instead.

    `offsets[cam]` is how far that camera's clock runs ahead of `ref`'s, the
    same convention `compute_offsets` returns.
    """
    ref_times, best = k_envelope(audio_by_cam[ref], cfg.rms_window_sec, sr)
    best = best.copy()
    for cam, path in audio_by_cam.items():
        if cam == ref:
            continue
        t, k = k_envelope(path, cfg.rms_window_sec, sr)
        # Outside this camera's recording there is nothing to hear, so fill
        # with a value that can never win the comparison below.
        aligned = np.interp(ref_times + offsets.get(cam, 0.0), t, k,
                            left=-np.inf, right=-np.inf)
        best = np.maximum(best, aligned)
    over = np.isfinite(best) & (best >= cfg.threshold_k)
    return _suppress(list(zip(ref_times[over], best[over])), cfg)
