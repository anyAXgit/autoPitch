from dataclasses import dataclass
import yaml


@dataclass
class Config:
    fps: int = 30
    build_up_sec: float = 5.0
    min_len_sec: float = 10.0
    max_len_sec: float = 25.0
    rms_window_sec: float = 0.5
    threshold_k: float = 3.0
    min_gap_sec: float = 15.0
    max_clips: "int | None" = None
    margin_db: float = 6.0
    hold_sec: float = 2.0
    crossfade_sec: float = 0.5
    output_width: int = 1920
    output_height: int = 1080


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    peak = raw.get("peak", {}) or {}
    reaction = raw.get("reaction", {}) or {}
    defaults = Config()
    return Config(
        fps=raw.get("fps", defaults.fps),
        build_up_sec=raw.get("build_up_sec", defaults.build_up_sec),
        min_len_sec=raw.get("min_len_sec", defaults.min_len_sec),
        max_len_sec=raw.get("max_len_sec", defaults.max_len_sec),
        rms_window_sec=peak.get("rms_window_sec", defaults.rms_window_sec),
        threshold_k=peak.get("threshold_k", defaults.threshold_k),
        min_gap_sec=peak.get("min_gap_sec", defaults.min_gap_sec),
        max_clips=peak.get("max_clips", defaults.max_clips),
        margin_db=reaction.get("margin_db", defaults.margin_db),
        hold_sec=reaction.get("hold_sec", defaults.hold_sec),
        crossfade_sec=raw.get("crossfade_sec", defaults.crossfade_sec),
        output_width=raw.get("output_width", defaults.output_width),
        output_height=raw.get("output_height", defaults.output_height),
    )
