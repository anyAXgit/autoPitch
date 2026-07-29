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
    min_confidence: float = 0.0   # keep a goal only if verdict confidence >= this (0 = off)


@dataclass
class LocateConfig:
    """Net-ROI goal localization (fixed cameras): find the exact goal frame from a
    net-motion spike inside a calibrated goal box, else fall back to cheer onset."""
    enabled: bool = False
    rois_path: "str | None" = None   # net_rois.json: {camKeySubstring: [x,y,w,h] normalized}
    # Search window [onset-pre, onset+post], biased BEFORE the onset: the ball hits
    # the net a beat before the crowd cheers, so a spike well after onset is
    # keeper-retrieval / scramble, not the goal.
    pre_sec: float = 2.5
    post_sec: float = 0.5
    fps: float = 10.0               # motion sampling rate in the window
    min_prominence: float = 6.0     # spike must exceed window median by this many MADs
    scan_enabled: bool = False      # add ROI-only candidates; expensive, run only when requested/cached
    scan_min_prominence: float = 10.0  # stricter threshold for ROI-only clip candidates
    scan_fps: float = 2.0           # cheaper full-game ROI-only scan rate
    scan_frame_px: int = 32         # cheaper full-game ROI-only frame size
    scan_max_impulse_sec: float = 1.2   # Tier-0 gate: net-hit motion must be a brief impulse
    scan_max_candidates: int = 40   # refine/gate only the strongest rough hits (cost bound)
    scan_verify: str = "shape"      # ROI-only verification: shape (free) | vlm (net-crop judge)
    scan_verify_model: str = "claude-haiku-4-5-20251001"  # cheap judge; ~3 crops/event
    scan_cache: str = "data/_gui/roi_scan_cache.json"   # whole-match scan results cache
    weak_audio_k: float = 2.0       # add ROI-verified candidates from sub-threshold audio peaks
    weak_audio_min_confidence: float = 40.0  # ROI confidence floor for weak-audio candidates
    weak_audio_max_lead_sec: float = 5.0  # ROI hit may precede weak audio by only a few seconds
    frame_px: int = 64              # ROI downscaled to frame_px^2 for the diff


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
    min_angle_switch_sec: float = 0.0
    tail_sec: float = 0.0
    crossfade_sec: float = 0.5
    output_width: int = 1920
    output_height: int = 1080
    hw_encode: bool = True   # use Apple VideoToolbox H.264 when available (~2x faster here)
    main_cam: "str | None" = None
    bgm_path: "str | None" = None    # optional background-music file mixed under highlight_all
    bgm_volume: float = 0.15
    vision: VisionConfig = field(default_factory=VisionConfig)
    locate: LocateConfig = field(default_factory=LocateConfig)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    peak = raw.get("peak", {}) or {}
    reaction = raw.get("reaction", {}) or {}
    vision = raw.get("vision", {}) or {}
    locate = raw.get("locate", {}) or {}
    defaults = Config()
    vdef = defaults.vision
    ldef = defaults.locate
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
        min_angle_switch_sec=reaction.get("min_angle_switch_sec", defaults.min_angle_switch_sec),
        tail_sec=reaction.get("tail_sec", defaults.tail_sec),
        crossfade_sec=raw.get("crossfade_sec", defaults.crossfade_sec),
        output_width=raw.get("output_width", defaults.output_width),
        output_height=raw.get("output_height", defaults.output_height),
        hw_encode=raw.get("hw_encode", defaults.hw_encode),
        main_cam=raw.get("main_cam", defaults.main_cam),
        bgm_path=raw.get("bgm_path", defaults.bgm_path),
        bgm_volume=raw.get("bgm_volume", defaults.bgm_volume),
        vision=VisionConfig(
            enabled=vision.get("enabled", vdef.enabled),
            pre_sec=vision.get("pre_sec", vdef.pre_sec),
            post_sec=vision.get("post_sec", vdef.post_sec),
            fps=vision.get("fps", vdef.fps),
            frame_height=vision.get("frame_height", vdef.frame_height),
            model=vision.get("model", vdef.model),
            min_confidence=vision.get("min_confidence", vdef.min_confidence),
        ),
        locate=LocateConfig(
            enabled=locate.get("enabled", ldef.enabled),
            rois_path=locate.get("rois_path", ldef.rois_path),
            pre_sec=locate.get("pre_sec", ldef.pre_sec),
            post_sec=locate.get("post_sec", ldef.post_sec),
            fps=locate.get("fps", ldef.fps),
            min_prominence=locate.get("min_prominence", ldef.min_prominence),
            scan_max_impulse_sec=locate.get("scan_max_impulse_sec", ldef.scan_max_impulse_sec),
            scan_max_candidates=locate.get("scan_max_candidates", ldef.scan_max_candidates),
            scan_verify=locate.get("scan_verify", ldef.scan_verify),
            scan_verify_model=locate.get("scan_verify_model", ldef.scan_verify_model),
            scan_cache=locate.get("scan_cache", ldef.scan_cache),
            scan_enabled=locate.get("scan_enabled", ldef.scan_enabled),
            scan_min_prominence=locate.get("scan_min_prominence", ldef.scan_min_prominence),
            scan_fps=locate.get("scan_fps", ldef.scan_fps),
            scan_frame_px=locate.get("scan_frame_px", ldef.scan_frame_px),
            weak_audio_k=locate.get("weak_audio_k", ldef.weak_audio_k),
            weak_audio_min_confidence=locate.get("weak_audio_min_confidence", ldef.weak_audio_min_confidence),
            weak_audio_max_lead_sec=locate.get("weak_audio_max_lead_sec", ldef.weak_audio_max_lead_sec),
            frame_px=locate.get("frame_px", ldef.frame_px),
        ),
    )
