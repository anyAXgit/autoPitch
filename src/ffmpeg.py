"""Locate ffmpeg/ffprobe and pick platform-appropriate encoders.

Everything in the pipeline shells out to ffmpeg, so a packaged build needs one
place that answers "which binary?". Resolution order is deliberate:

  1. AUTOPITCH_FFMPEG / AUTOPITCH_FFPROBE  -- explicit override, wins always
  2. the copy shipped next to a frozen build -- a user's ancient Homebrew ffmpeg
     must not silently replace the build we tested against
  3. PATH -- the source-checkout case

Hardware encoding is probed, not assumed: several encoders are *listed* by
`ffmpeg -encoders` on machines where creating a session fails (headless macOS,
a laptop whose dGPU is off), and discovering that only when a 20-minute render
dies is not acceptable. Each candidate gets a 320x180 encode first.
"""
import os
import shutil
import subprocess
import sys

_INSTALL_HINT = {
    "darwin": "brew install ffmpeg",
    "win32": "winget install --id Gyan.FFmpeg",
}.get(sys.platform, "sudo apt install ffmpeg")


class FFmpegMissing(RuntimeError):
    """ffmpeg (or ffprobe) could not be found on this machine."""


_CACHE = {}

# An app launched from Finder does not inherit the shell's PATH -- it gets a
# bare `/usr/bin:/bin:/usr/sbin:/sbin`. Homebrew installs to /opt/homebrew/bin
# (Apple silicon) or /usr/local/bin (Intel), neither of which is in that list.
# So `which("brew")` and `which("ffmpeg")` both come back empty on a machine
# where the user can run either one from a terminal -- and the app tells them to
# install what they already installed. Look in the standard places too.
_EXTRA_BIN_DIRS = [
    "/opt/homebrew/bin",                    # Homebrew, Apple silicon
    "/usr/local/bin",                       # Homebrew on Intel; manual installs
    "/opt/local/bin",                       # MacPorts
    "/snap/bin",                            # Linux, snap
    r"C:\ProgramData\chocolatey\bin",       # Windows, Chocolatey
]


def search_path():
    """PATH plus the install locations a GUI launch cannot see.

    Also handed to subprocesses: `brew install` needs to find its own tools, and
    inheriting our stripped-down PATH is what breaks it.
    """
    dirs = os.environ.get("PATH", "").split(os.pathsep)
    dirs += [d for d in _EXTRA_BIN_DIRS
             if os.path.isdir(d) and d not in dirs]
    return os.pathsep.join(d for d in dirs if d)


def which(name):
    """`shutil.which`, widened to the standard install locations."""
    return shutil.which(name, path=search_path())


def _bundled_dir():
    """Where a PyInstaller build keeps its binaries, or None in a checkout."""
    if not getattr(sys, "frozen", False):
        return None
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    return os.path.join(base, "bin")


def _resolve(name):
    exe = name + (".exe" if os.name == "nt" else "")
    override = os.environ.get(f"AUTOPITCH_{name.upper()}")
    if override:
        if not os.path.exists(override):
            raise FFmpegMissing(
                f"AUTOPITCH_{name.upper()}가 가리키는 파일이 없습니다: {override}")
        return override
    bundled = _bundled_dir()
    if bundled:
        cand = os.path.join(bundled, exe)
        if os.path.exists(cand):
            return cand
    found = which(name)
    if found:
        return found
    raise FFmpegMissing(
        f"{name}을(를) 찾을 수 없습니다. 설치 후 다시 실행해 주세요:\n"
        f"    {_INSTALL_HINT}\n"
        f"(이미 설치했다면 AUTOPITCH_{name.upper()} 환경변수로 경로를 지정할 수 있습니다.)")


def ffmpeg():
    if "ffmpeg" not in _CACHE:
        _CACHE["ffmpeg"] = _resolve("ffmpeg")
    return _CACHE["ffmpeg"]


def ffprobe():
    if "ffprobe" not in _CACHE:
        _CACHE["ffprobe"] = _resolve("ffprobe")
    return _CACHE["ffprobe"]


def available():
    """(ok, message) -- for the first-run check, which must not raise."""
    try:
        out = subprocess.run([ffmpeg(), "-version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
        return True, out
    except FFmpegMissing as e:
        return False, str(e)
    except Exception as e:                      # present but not runnable
        return False, f"ffmpeg 실행에 실패했습니다: {e}"


def run(args, **kw):
    """`ffmpeg -y -hide_banner -loglevel error <args>`, raising on failure."""
    kw.setdefault("check", True)
    return subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                           *args], **kw)


def probe(args, **kw):
    kw.setdefault("check", True)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run([ffprobe(), "-v", "error", *args], **kw)


# --- frame-rate mode ---------------------------------------------------------
# `-vsync` was deprecated in ffmpeg 5.1 in favour of `-fps_mode`, warned about
# for years, and is gone in 9.x -- where it is not a warning but a failed
# command. We cannot pick one spelling: the user brings their own ffmpeg, and
# both an Ubuntu LTS 4.x and a brand-new 9.x are things people actually have.
# So ask this ffmpeg, once, the same way the encoder probe does.
_FPS_MODE = {"cfr": "cfr", "0": "passthrough"}


def fps_mode_args(mode):
    """`-fps_mode <x>` on ffmpeg that knows it, `-vsync <x>` on ffmpeg that
    doesn't. `mode` is the old -vsync spelling ("cfr" or "0")."""
    if "fps_mode" not in _CACHE:
        _CACHE["fps_mode"] = subprocess.run([
            ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=32x32:d=0.1:r=30",
            "-fps_mode", "cfr", "-frames:v", "1", "-f", "null", "-",
        ], capture_output=True, text=True).returncode == 0
    if _CACHE["fps_mode"]:
        return ["-fps_mode", _FPS_MODE[mode]]
    return ["-vsync", mode]


# --- hardware ---------------------------------------------------------------
# (encoder, extra args). Ordered by platform likelihood; the first that both
# lists AND survives a probe encode wins.
_HW_CANDIDATES = [
    ("h264_videotoolbox", ["-b:v", "10M", "-allow_sw", "1"]),   # Apple
    ("h264_nvenc", ["-b:v", "10M", "-preset", "p4"]),           # NVIDIA
    ("h264_qsv", ["-b:v", "10M"]),                              # Intel
    ("h264_amf", ["-b:v", "10M"]),                              # AMD
]


def _hw_encoder():
    """First H.264 encoder that actually encodes here, or None."""
    if "hw" in _CACHE:
        return _CACHE["hw"]
    _CACHE["hw"] = None
    try:
        listed = subprocess.run([ffmpeg(), "-hide_banner", "-encoders"],
                                capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    for enc, extra in _HW_CANDIDATES:
        if enc not in listed:
            continue
        probe_ok = subprocess.run([
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x180:d=0.1:r=30",
            "-frames:v", "1", "-c:v", enc, *extra, "-f", "null", "-",
        ], capture_output=True, text=True).returncode == 0
        if probe_ok:
            _CACHE["hw"] = (enc, extra)
            break
    return _CACHE["hw"]


def h264_args(hw=True):
    """Encoder args for H.264 output.

    Hardware encoding measured ~2x faster than libx264 on this workload (HEVC
    *decode* dominates the rest). Falls back to libx264 when no usable hardware
    encoder exists or when the config turns it off.
    """
    if hw:
        found = _hw_encoder()
        if found:
            enc, extra = found
            return ["-c:v", enc, *extra, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p"]


def hwaccel_args():
    """Decode acceleration flags, empty when the platform has none to offer."""
    if sys.platform == "darwin":
        return ["-hwaccel", "videotoolbox"]
    if sys.platform == "win32":
        return ["-hwaccel", "d3d11va"]
    return []


def concat_path(path):
    """Absolute path formatted for the concat demuxer.

    Inside the quotes a backslash is an escape character, so a Windows path
    written verbatim (`file 'C:\\clips\\a.mp4'`) is silently mangled. ffmpeg
    accepts forward slashes on every platform, so normalise instead of escaping.
    """
    return os.path.abspath(path).replace("\\", "/").replace("'", r"'\''")
