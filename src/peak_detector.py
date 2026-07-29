import numpy as np

from src.audio import block_rms, load


def rms_db(path, window_sec, sr=16000):
    y = load(path, sr)
    hop = max(1, int(window_sec * sr))
    rms = block_rms(y, hop)
    db = 20.0 * np.log10(rms + 1e-8)
    times = np.arange(len(db)) * hop / sr
    return times, db


def detect_peaks(path, cfg, sr=16000):
    times, db = rms_db(path, cfg.rms_window_sec, sr)
    threshold = np.median(db) + cfg.threshold_k * np.std(db)
    cand = [(times[i], db[i]) for i in range(len(db)) if db[i] >= threshold]
    if not cand:
        return []
    # suppress within min_gap: keep loudest per cluster
    cand.sort(key=lambda c: c[0])
    kept = []
    for t, d in cand:
        if kept and t - kept[-1][0] < cfg.min_gap_sec:
            # Round before comparing so sub-0.1dB float/codec noise on an
            # otherwise flat burst plateau doesn't let a later frame "win"
            # by a razor-thin margin; ties resolve to the earlier (onset)
            # frame, which is the semantically correct goal moment.
            if round(d, 1) > round(kept[-1][1], 1):
                kept[-1] = (t, d)
        else:
            kept.append((t, d))
    if cfg.max_clips is not None and len(kept) > cfg.max_clips:
        kept = sorted(kept, key=lambda c: c[1], reverse=True)[: cfg.max_clips]
    return sorted(t for t, _ in kept)
