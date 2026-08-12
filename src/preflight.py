"""Environment checks shown on the first-run screen.

A first-time user's failures are almost never in the pipeline -- they are a
missing ffmpeg, a stripped ffmpeg build without xfade, or a folder the app
cannot write to. Those used to surface as a stack trace twenty minutes into a
render. Each check here returns a plain-language fix, and nothing raises: the
setup screen must render even when everything is broken.
"""
import os
import platform
import shutil
import subprocess
import sys


def _hint(pkg_mac, pkg_win, pkg_linux):
    return {"darwin": pkg_mac, "win32": pkg_win}.get(sys.platform, pkg_linux)


def check_ffmpeg():
    from src.ffmpeg import available, ffmpeg
    ok, msg = available()
    if not ok:
        return {"id": "ffmpeg", "ok": False, "title": "ffmpeg",
                "desc": "영상을 자르고 합치려면 ffmpeg가 필요합니다.",
                "fix": _hint("brew install ffmpeg",
                             "winget install --id Gyan.FFmpeg",
                             "sudo apt install ffmpeg")}
    ver = msg.split(" ")[2] if len(msg.split(" ")) > 2 else "?"
    return {"id": "ffmpeg", "ok": True, "title": "ffmpeg",
            "desc": f"{ver} · {ffmpeg()}"}


def check_filters():
    """xfade/acrossfade drive the angle-cut crossfade; minimal builds omit them."""
    from src.ffmpeg import ffmpeg, FFmpegMissing
    try:
        out = subprocess.run([ffmpeg(), "-hide_banner", "-filters"],
                             capture_output=True, text=True, check=True).stdout
    except FFmpegMissing:
        return None                        # ffmpeg check already reported it
    except Exception as e:
        return {"id": "filters", "ok": False, "title": "ffmpeg 기능",
                "desc": f"필터 목록을 읽지 못했습니다: {e}", "fix": ""}
    missing = [f for f in ("xfade", "acrossfade") if f not in out]
    if missing:
        return {"id": "filters", "ok": False, "title": "ffmpeg 기능",
                "desc": f"{', '.join(missing)} 필터가 없는 축소 빌드입니다. 앵글 전환 렌더가 실패합니다.",
                "fix": _hint("brew install ffmpeg", "winget install --id Gyan.FFmpeg",
                             "sudo apt install ffmpeg")}
    return {"id": "filters", "ok": True, "title": "ffmpeg 기능",
            "desc": "앵글 전환에 필요한 필터가 모두 있습니다."}


def check_encoder():
    """Informational -- libx264 always works, hardware is just faster."""
    try:
        from src.ffmpeg import _hw_encoder
        found = _hw_encoder()
    except Exception:
        found = None
    if found:
        return {"id": "encoder", "ok": True, "title": "하드웨어 인코딩",
                "desc": f"{found[0]} 사용 · 소프트웨어 대비 약 2배 빠릅니다."}
    return {"id": "encoder", "ok": True, "warn": True, "title": "하드웨어 인코딩",
            "desc": "사용 가능한 GPU 인코더가 없어 libx264로 렌더합니다. 느리지만 정상 동작합니다.",
            "fix": ""}


def check_writable(root):
    probe = os.path.join(root, "data", ".write_test")
    try:
        os.makedirs(os.path.dirname(probe), exist_ok=True)
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as e:
        return {"id": "write", "ok": False, "title": "폴더 권한",
                "desc": f"{root} 에 쓸 수 없습니다: {e}",
                "fix": "문서 폴더처럼 쓰기 가능한 위치를 프로젝트 폴더로 지정하세요."}
    return {"id": "write", "ok": True, "title": "폴더 권한",
            "desc": "작업 폴더에 저장할 수 있습니다."}


def check_disk(root, need_gb=10):
    try:
        free = shutil.disk_usage(root).free / 1e9
    except Exception:
        return None
    if free < need_gb:
        return {"id": "disk", "ok": False, "warn": True, "title": "디스크 여유 공간",
                "desc": f"{free:.0f}GB 남았습니다. 2시간 분량을 처리하려면 {need_gb}GB 이상 권장합니다.",
                "fix": "공간을 확보하거나 출력 폴더를 다른 드라이브로 지정하세요."}
    return {"id": "disk", "ok": True, "title": "디스크 여유 공간",
            "desc": f"{free:.0f}GB 사용 가능."}


def check_vision_key(root):
    """Optional -- the pipeline runs fine without it, so never a hard failure.

    Checks the package as well as the key: having one without the other fails
    only once a render is already under way, and this screen is where that is
    still cheap to notice.
    """
    have = bool(os.environ.get("ANTHROPIC_API_KEY")) or os.path.exists(
        os.path.join(root, "data", "_gui", "anthropic_key.txt"))
    if have:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return {"id": "vision", "ok": True, "warn": True,
                    "title": "AI 골 라벨링 (선택)",
                    "desc": "API 키는 있는데 anthropic 패키지가 없습니다. 켜면 그 단계에서 멈춥니다.",
                    "fix": "pip install anthropic"}
        return {"id": "vision", "ok": True, "title": "AI 골 라벨링 (선택)",
                "desc": "API 키를 찾았습니다. 후보에 골 여부를 기록합니다."}
    return {"id": "vision", "ok": True, "warn": True, "title": "AI 골 라벨링 (선택)",
            "desc": "키가 없어도 하이라이트는 정상적으로 만들어집니다. 켜면 후보마다 골 여부가 데이터로 기록됩니다.",
            "fix": "data/_gui/anthropic_key.txt 에 API 키를 저장하면 자동으로 인식합니다."}


def run(root):
    """All checks. `blocking` is what actually stops the user."""
    checks = [c for c in (check_ffmpeg(), check_filters(), check_encoder(),
                          check_writable(root), check_disk(root),
                          check_vision_key(root)) if c]
    return {
        "checks": checks,
        "blocking": [c["id"] for c in checks if not c["ok"]],
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
