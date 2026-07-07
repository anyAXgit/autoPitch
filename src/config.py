from dataclasses import dataclass, field
import yaml


@dataclass
class VisionConfig:
    """Sparse-frame goal-confirmation settings (V2/V3). Disabled by default so
    the audio-only pipeline is unchanged unless a `vision:` block opts in."""
    enabled: bool = False
    pre_sec: float = 3.0      # frames from T-pre_sec (covers the shot, ~1-2s before the cheer peak)
    post_sec: float = 5.0     # ... to T+post_sec (covers ball-in + walk-back to restart)
    fps: float = 2.0          # sampling rate within that window
    frame_height: int = 360   # JPG height (width auto, aspect-preserved)
    model: str = "claude-opus-4-8"   # VLM used to judge "goal celebration?"


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
    post_goal_sec: float = 2.5
    min_reaction_sec: float = 2.0
    tail_sec: float = 0.0
    crossfade_sec: float = 0.5
    output_width: int = 1920
    output_height: int = 1080
    main_cam: "str | None" = None
    vision: VisionConfig = field(default_factory=VisionConfig)


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    peak = raw.get("peak", {}) or {}
    reaction = raw.get("reaction", {}) or {}
    vision = raw.get("vision", {}) or {}
    defaults = Config()
    vdef = defaults.vision
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
        post_goal_sec=reaction.get("post_goal_sec", defaults.post_goal_sec),
        min_reaction_sec=reaction.get("min_reaction_sec", defaults.min_reaction_sec),
        tail_sec=reaction.get("tail_sec", defaults.tail_sec),
        crossfade_sec=raw.get("crossfade_sec", defaults.crossfade_sec),
        output_width=raw.get("output_width", defaults.output_width),
        output_height=raw.get("output_height", defaults.output_height),
        main_cam=raw.get("main_cam", defaults.main_cam),
        vision=VisionConfig(
            enabled=vision.get("enabled", vdef.enabled),
            pre_sec=vision.get("pre_sec", vdef.pre_sec),
            post_sec=vision.get("post_sec", vdef.post_sec),
            fps=vision.get("fps", vdef.fps),
            frame_height=vision.get("frame_height", vdef.frame_height),
            model=vision.get("model", vdef.model),
        ),
    )
