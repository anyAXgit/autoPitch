"""Vision goal-labelling seam.

A Claude VLM is shown the sparse per-goal frames from frame_extractor and asked
whether a goal was scored. The classifier is injectable so tests can stub it.

`label_goals` is what the pipeline uses: it records each verdict as data and
drops nothing. Loud non-goals -- saves, near misses -- are wanted content in a
highlight reel, and the model can be wrong, so what to keep stays a human
decision in the editor.

`confirm_goals` is the older path that prunes peaks instead. Nothing calls it
but its own tests; it is kept as the seam for a pipeline that wants precision
over recall.
"""
import base64
import json
import os
from src.errors import UserError
from src.ffmpeg import ffmpeg

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


def label_goals(peaks, frames_by_T, cfg, classifier=None):
    """Return {peak_time: {"is_goal": bool, "confidence": float}} for every peak.

    Nothing is dropped and nothing is drawn. The product is a HIGHLIGHT reel, so
    loud non-goals (saves, near misses) are wanted content; and an audio
    candidate is not guaranteed to be a goal, so burning a "GOAL" caption on the
    model's guess would publish its mistakes. The verdict is therefore recorded
    as DATA only -- it lands in plan.json for review, sorting and future
    training. Whether a clip actually gets a badge stays a manual editor choice
    (`goal_label`), which this never touches.

    Returns an empty dict when vision is disabled or no classifier is supplied.
    """
    if not cfg.vision.enabled or classifier is None:
        return {}
    labels = {}
    for T in peaks:
        frames = frames_by_T.get(T) or []
        if not frames:
            labels[float(T)] = {"is_goal": False, "confidence": 0.0}
            continue
        v = classifier(frames) or {}
        conf = float(v.get("confidence", 0.0))
        ok = bool(v.get("is_goal")) and conf >= cfg.vision.min_confidence
        labels[float(T)] = {"is_goal": ok, "confidence": conf}
    return labels


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
        if verdict.get("is_goal") and verdict.get("confidence", 1.0) >= cfg.vision.min_confidence:
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


_NET_PROMPT = (
    "These are close-up crops of a futsal GOAL NET from a fixed camera around a "
    "candidate goal moment. Most frames were auto-selected because a ball-like "
    "white blob appeared inside the net area — your job is to verify it. Decide "
    "if a GOAL was scored: a ball LYING/RESTING INSIDE the goal/net, the ball "
    "entering the net, or the mesh bulging from a ball impact. Careful: the "
    "auto-selector confuses white goal-post edges, pad edges, and white jerseys "
    "seen through the mesh for a ball — those are NOT goals. Also not a goal: a "
    "keeper handling the net, someone fetching a ball from beside the goal, a "
    "ball on the FIELD outside the goal mouth, or nothing notable.\n"
    'Reply with ONLY a JSON object: {"is_goal": true|false, "confidence": 0.0-1.0}.'
)

# The verdict is ~40 tokens, but current models think before answering and
# `max_tokens` caps thinking and reply together. 1024 was enough on a model where
# thinking stayed off unless asked for; on one where it is on by default the reply
# can be cut off before its closing brace -- which `_verdict` cannot tell apart
# from a model that answered in prose. Headroom costs nothing: only generated
# tokens are billed.
_MAX_TOKENS = 4096


def _client():
    """An Anthropic client, or a sentence saying how to get one.

    `anthropic` is an optional extra -- the pipeline runs without it and only
    this step needs it -- so a plain ImportError here reads like a broken
    install rather than a feature the user has not opted into yet.
    """
    try:
        import anthropic
    except ImportError:
        raise UserError(
            "골 라벨링에는 anthropic 패키지가 필요합니다.\n"
            "  설치: pip install anthropic\n"
            "  끄려면 config.yaml 의 vision.enabled 를 false 로 두세요.") from None
    return anthropic.Anthropic()


def _verdict(resp):
    """Read {"is_goal", "confidence"} out of a reply.

    Anything unreadable -- truncated, prose-wrapped, refused -- keeps the clip at
    zero confidence. Dropping a real goal is the expensive mistake; a clip that
    should not be there is one the editor removes in a click.
    """
    if resp.stop_reason == "max_tokens":
        return {"is_goal": True, "confidence": 0.0}
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
        return {"is_goal": bool(data.get("is_goal")),
                "confidence": float(data.get("confidence", 0.0))}
    except (ValueError, json.JSONDecodeError):
        return {"is_goal": True, "confidence": 0.0}


def net_crops(source, t, roi, out_dir, margin=0.6):
    """Extract net close-up JPGs around `t` (source timeline) for the Tier-1
    VLM judge. Frame moments are BALL-GUIDED: a dense scan picks the frames most
    likely to show a ball inside the net (the ball rests there at a variable
    moment 0-5s after impact — fixed offsets miss it; measured recall 1/5).
    Falls back to fixed offsets when nothing ball-like is seen."""
    import subprocess
    from src.ball_frames import select_ball_times
    x, y, w, h = roi
    x0 = max(0.0, x - w * margin)
    y0 = max(0.0, y - h * margin)
    x1 = min(1.0, x + w * (1 + margin))
    y1 = min(1.0, y + h * (1 + margin))
    vf = (f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f},"
          f"scale=-2:360")
    os.makedirs(out_dir, exist_ok=True)
    try:
        times = select_ball_times(source, t, roi)
    except Exception:
        times = []
    # always include one impact-context frame; ball-guided picks carry the evidence
    moments = [max(0.0, t + 0.2)] + times
    if not times:   # fallback: spread fixed offsets across the settle window
        moments += [max(0.0, t + dt) for dt in (-0.5, 1.0, 2.0, 3.5)]
    paths = []
    for i, ts in enumerate(moments):
        p = os.path.join(out_dir, f"net_{t:.1f}_{i}.jpg")
        subprocess.run([ffmpeg(), "-y", "-v", "error", "-ss", str(ts),
                        "-i", source, "-frames:v", "1", "-vf", vf, p], check=True)
        paths.append(p)
    return paths


def make_net_classifier(cfg, client=None):
    """Tier-1 judge for ROI-only candidates: sends 3 net close-ups to a VLM and
    asks specifically about a ball hitting/entering the net (not celebrations --
    the crop IS the net, so the question can be concrete)."""
    client = client or _client()

    def classify(crop_paths):
        if not crop_paths:
            return {"is_goal": True, "confidence": 0.0}
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": _encode(p)}}
                   for p in crop_paths]
        content.append({"type": "text", "text": _NET_PROMPT})
        resp = client.messages.create(
            model=cfg.locate.scan_verify_model or cfg.vision.model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
        return _verdict(resp)

    return classify


def confirm_roi_clips(plan, cfg, rois, offsets, classifier, workdir):
    """Filter the plan's ROI-only clips (quiet-goal candidates with no audio
    confirmation) through a net-crop classifier. Audio-backed clips pass through
    untouched; ROI-only clips get a `roi_verdict` and are dropped when the judge
    says not-a-goal. Fail-open on missing ROI/frames (keep the clip flagged)."""
    from src import goal_locator
    kept = []
    for clip in plan["clips"]:
        if not clip.get("roi_only"):
            kept.append(clip)
            continue
        cam = clip.get("goal_cam") or clip["segments"][0]["cam"]
        src = next((g["src"] for g in clip["segments"] if g["cam"] == cam),
                   clip["segments"][0]["src"])
        roi = goal_locator.roi_for_cam(cam, rois, src)
        if roi is None:
            kept.append(clip)
            continue
        t = clip["T"] + offsets.get(cam, 0.0)
        try:
            crops = net_crops(src, t, roi, workdir)
            verdict = classifier(crops)
        except Exception:  # noqa: keep the candidate rather than crash planning
            kept.append(clip)
            continue
        clip["roi_verdict"] = verdict
        if verdict.get("is_goal"):
            kept.append(clip)
        else:
            # judged-not-a-goal events are the most valuable NEGATIVE labels for
            # the future local model -- surface them so the caller can log them.
            plan.setdefault("roi_rejected", []).append(clip)
    plan["clips"] = kept
    return plan


def make_vlm_classifier(cfg, client=None, max_frames=10):
    """Build a classifier(frames)->{"is_goal": bool, "confidence": float} backed by
    a Claude VLM. `client` is injectable; by default an anthropic.Anthropic() is
    created lazily (needs credentials). Frames are subsampled to `max_frames` to
    bound tokens/cost."""
    client = client or _client()

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
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
        return _verdict(resp)

    return classify
