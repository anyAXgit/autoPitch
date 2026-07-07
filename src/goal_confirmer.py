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
    "These are consecutive frames (a few seconds) from one fixed low corner "
    "camera at an indoor 5-a-side futsal match, around a moment the crowd got "
    "loud. Decide whether a GOAL was most likely just scored. Celebrations are "
    "usually MINIMAL in casual play, so do NOT require an obvious celebration. "
    "Treat ANY of these as goal-likely:\n"
    "- the ball going into / hitting a goal net, or a clear shot on a goal\n"
    "- players converging near a goal, or a keeper retrieving the ball from the net\n"
    "- players jogging or walking back toward the center circle to restart "
    "(a kickoff right after a goal is the most reliable cue here)\n"
    "- any celebration: arms raised, a small cluster, pointing, high-fives\n"
    "It is NOT a goal only if the frames show ordinary open play in midfield "
    "with no shot, no ball near a net, and no restart.\n"
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


def _subsample(frames, n):
    """Evenly-spaced subsample of `frames` down to at most `n`, covering the whole
    window (not just the front)."""
    if len(frames) <= n:
        return list(frames)
    if n <= 1:
        return [frames[0]]
    return [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)]


def make_vlm_classifier(cfg, client=None, max_frames=10):
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
        picked = _subsample(frames, max_frames)   # even coverage of the window
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
