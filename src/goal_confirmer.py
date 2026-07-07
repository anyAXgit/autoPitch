"""Vision goal-confirmation seam (V3).

`confirm_goals` filters audio-detected peaks down to those a classifier judges to
be real goal celebrations. The classifier is injectable so tests can stub it; the
default real classifier (`make_vlm_classifier`) asks a Claude VLM about the sparse
per-goal frames from frame_extractor.

Design: this only ever *prunes* peaks (raises precision). Recall is raised
separately by lowering `peak.threshold_k` so audio over-proposes and this filter
removes the junk. A goal with no extracted frames is kept (don't drop on missing
evidence).
"""
import base64
import json
import os

_PROMPT = (
    "These frames are consecutive moments from one camera at an indoor futsal "
    "match, spanning a few seconds around a loud crowd cheer. Did a GOAL just "
    "happen? Look for goal-celebration cues: players running to celebrate, arms "
    "raised, a cluster forming, the ball in or near the net. A loud moment with "
    "normal open play (no celebration) is NOT a goal.\n"
    'Reply with ONLY a JSON object: {"is_goal": true|false, "confidence": 0.0-1.0}.'
)


def confirm_goals(peaks, frames_by_T, cfg, classifier=None):
    """Return the subset of `peaks` confirmed as goals.

    Passthrough (returns peaks unchanged) when vision is disabled or no classifier
    is supplied, so the audio-only pipeline is unaffected by default.
    `frames_by_T` maps a peak time to its extracted frame paths.
    """
    if not cfg.vision.enabled or classifier is None:
        return list(peaks)
    kept = []
    for T in peaks:
        frames = frames_by_T.get(T) or []
        if not frames:
            kept.append(T)          # no evidence to judge on -> keep
            continue
        verdict = classifier(frames)
        if verdict.get("is_goal"):
            kept.append(T)
    return kept


def _encode(frame_path):
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def make_vlm_classifier(cfg, client=None, max_frames=8):
    """Build a classifier(frames)->{"is_goal": bool, "confidence": float} backed by
    a Claude VLM. `client` is injectable; by default an anthropic.Anthropic() is
    created lazily (needs credentials). Frames are subsampled to `max_frames` to
    bound tokens/cost."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    def classify(frames):
        if not frames:
            return {"is_goal": True, "confidence": 0.0}
        # even subsample down to max_frames, keeping temporal order
        step = max(1, len(frames) // max_frames)
        picked = frames[::step][:max_frames]
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": _encode(fp)}}
            for fp in picked
        ]
        content.append({"type": "text", "text": _PROMPT})
        resp = client.messages.create(
            model=cfg.vision.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
            return {"is_goal": bool(data.get("is_goal")),
                    "confidence": float(data.get("confidence", 0.0))}
        except (ValueError, json.JSONDecodeError):
            # Unparseable response: keep the goal rather than silently dropping it.
            return {"is_goal": True, "confidence": 0.0}

    return classify
