#!/usr/bin/env python3
"""Build the macOS .app / Windows .exe.

    python packaging/build.py                      # 현재 플랫폼용 빌드
    python packaging/build.py --sign               # + Developer ID 서명
    python packaging/build.py --sign --notarize    # + 애플 공증·스테이플
    python packaging/build.py --vendor-ffmpeg /path/to/ffmpeg /path/to/ffprobe

PyInstaller cannot cross-compile: run this on macOS for the .app and on Windows
for the .exe. Output lands in `dist/`.

Signing happens here rather than through the spec's `codesign_identity`, because
order matters and the spec cannot express it: every Mach-O file inside the bundle
has to be signed before the bundle that contains it.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.console import enable_utf8  # noqa: E402

enable_utf8()
VENDOR = os.path.join(HERE, "vendor", "bin")
# Keep entitlements.plist free of XML comments. `plutil -lint` accepts them, but
# the parser codesign hands the file to is a different, stricter one and answers
# "AMFIUnserializeXML: syntax error near line 4" -- a lint-clean file that cannot
# be signed with. Explanations for the two entries live here instead:
#   allow-unsigned-executable-memory  CPython writes and runs memory it just
#       generated (ctypes closures, and any run-time codegen path).
#   disable-library-validation        the interpreter dlopen()s extension modules
#       from inside the bundle at run time.
# Both are hardened-runtime protections switched off, so add a third only with a
# crash that proves it necessary.
ENTITLEMENTS = os.path.join(HERE, "entitlements.plist")
NOTARY_PROFILE = os.environ.get("AUTOPITCH_NOTARY_PROFILE", "autopitch-notary")

# Mach-O and universal-binary magic, both byte orders. Reading four bytes beats
# shelling out to `file` once per entry -- a frozen app holds a few hundred.
_MACHO_MAGIC = {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",     # 32-bit
                b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",     # 64-bit
                b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}     # universal


class BuildError(SystemExit):
    """Something the operator has to fix. Printed as a sentence, not a traceback."""


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, **kw)


def vendor(paths):
    """Copy ffmpeg/ffprobe into the build so the app ships self-contained."""
    os.makedirs(VENDOR, exist_ok=True)
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"파일이 없습니다: {p}")
        dst = os.path.join(VENDOR, os.path.basename(p))
        shutil.copy2(p, dst)
        os.chmod(dst, 0o755)
        print(f"  번들 포함: {os.path.basename(p)}")
    print("\n⚠️  libx264 가 포함된 ffmpeg 빌드는 GPL 입니다. 배포한다면 해당\n"
          "   라이선스 전문과 소스 제공 의무를 함께 지켜야 합니다.\n")


def resolve_identity(explicit):
    """The Developer ID to sign with: the flag, then the env var, then the one
    installed identity. Guessing between several would sign a release with the
    wrong key, so more than one is an error the operator resolves."""
    if explicit:
        return explicit
    if os.environ.get("AUTOPITCH_SIGN_IDENTITY"):
        return os.environ["AUTOPITCH_SIGN_IDENTITY"]
    out = subprocess.run(["security", "find-identity", "-v", "-p", "codesigning"],
                         capture_output=True, text=True).stdout
    names = {line.split('"')[1] for line in out.splitlines()
             if '"Developer ID Application' in line}
    if not names:
        raise BuildError(
            "Developer ID Application 인증서가 없습니다.\n"
            "  발급 후 키체인에 설치하고 `security find-identity -v -p codesigning`"
            " 로 확인해 주세요.")
    if len(names) > 1:
        raise BuildError("Developer ID 인증서가 여러 개입니다. --sign 으로 지정해 주세요:\n  "
                         + "\n  ".join(sorted(names)))
    return names.pop()


def _machos(app):
    """Every Mach-O file in the bundle, deepest first.

    Signing seals what a directory contains, so a binary signed after the bundle
    around it invalidates that bundle's signature. Depth order is what makes the
    nested code valid at the moment the outer signature is taken.
    """
    found = []
    for root, _, files in os.walk(app):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                continue
            try:
                with open(p, "rb") as fh:
                    if fh.read(4) in _MACHO_MAGIC:
                        found.append(p)
            except OSError:
                continue
    return sorted(found, key=lambda p: p.count(os.sep), reverse=True)


def sign(app, identity):
    """Sign nested code, then the bundle.

    Entitlements go on the bundle alone -- macOS reads them from the main
    executable, and attaching them to a library grants nothing while making the
    signature harder to reason about.
    """
    base = ["codesign", "--force", "--timestamp", "--options", "runtime",
            "--sign", identity]
    inner = _machos(app)
    print(f"서명 중: 내부 바이너리 {len(inner)}개 → 번들")
    for p in inner:
        run(base + [p], capture_output=True)
    run(base + ["--entitlements", ENTITLEMENTS, app], capture_output=True)

    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app],
        capture_output=True)
    print("  서명 검증 통과")


def notarize(app, profile=NOTARY_PROFILE):
    """Submit to Apple, wait for the verdict, staple it into the bundle.

    Stapling is what makes the app open on a machine that is offline or behind a
    filter: without it Gatekeeper has to ask Apple at launch time.
    """
    zip_path = app + ".notarize.zip"
    # ditto, not zip: the bundle's symlinks and permission bits have to survive.
    run(["ditto", "-c", "-k", "--keepParent", app, zip_path])
    try:
        print(f"공증 제출 (프로필 {profile}) — 보통 1~5분 걸립니다")
        p = subprocess.run(
            ["xcrun", "notarytool", "submit", zip_path,
             "--keychain-profile", profile, "--wait", "--output-format", "json"],
            capture_output=True, text=True)
        try:
            res = json.loads(p.stdout or "{}")
        except json.JSONDecodeError:
            res = {}
        if res.get("status") != "Accepted":
            sid = res.get("id")
            detail = p.stderr.strip() or p.stdout.strip()
            if sid:
                log = subprocess.run(
                    ["xcrun", "notarytool", "log", sid, "--keychain-profile", profile],
                    capture_output=True, text=True).stdout
                detail = log.strip() or detail
            raise BuildError(f"공증 거부됨 ({res.get('status', '알 수 없음')}):\n{detail}")
        print(f"  공증 승인 — {res.get('id')}")
    finally:
        os.remove(zip_path)

    run(["xcrun", "stapler", "staple", app], capture_output=True)
    # The real question is not "did codesign succeed" but "will Gatekeeper open
    # it on someone else's Mac", and that is what spctl answers.
    out = subprocess.run(["spctl", "--assess", "--type", "exec", "-vv", app],
                         capture_output=True, text=True)
    verdict = (out.stderr + out.stdout).strip()
    if "accepted" not in verdict:
        raise BuildError(f"Gatekeeper 거부:\n{verdict}")
    print("  스테이플 완료 · Gatekeeper 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-ffmpeg", nargs="+", metavar="BIN",
                    help="번들에 포함할 ffmpeg/ffprobe 경로")
    ap.add_argument("--clean", action="store_true", help="build/ dist/ 먼저 삭제")
    ap.add_argument("--sign", nargs="?", const=True, metavar="IDENTITY",
                    help="Developer ID 로 서명 (생략 시 설치된 인증서 자동 선택)")
    ap.add_argument("--notarize", action="store_true",
                    help="애플에 공증 제출 후 스테이플 (--sign 필요)")
    a = ap.parse_args()

    if (a.sign or a.notarize) and platform.system() != "Darwin":
        raise BuildError("서명·공증은 macOS 에서만 가능합니다.")
    if a.notarize and not a.sign:
        raise BuildError("공증은 서명된 앱에만 됩니다. --sign 을 함께 주세요.")

    if a.vendor_ffmpeg:
        vendor(a.vendor_ffmpeg)
    if a.clean:
        for d in ("build", "dist"):
            shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           os.path.join(HERE, "autopitch.spec")]
    print("빌드 중:", platform.system(), platform.machine())
    subprocess.run(cmd, cwd=ROOT, check=True)

    dist = os.path.join(ROOT, "dist")
    app = os.path.join(dist, "autoPitch.app")
    if a.sign:
        if not os.path.isdir(app):
            raise BuildError(f"서명할 앱이 없습니다: {app}")
        identity = resolve_identity(a.sign if isinstance(a.sign, str) else None)
        print(f"\n인증서: {identity}")
        sign(app, identity)
        if a.notarize:
            notarize(app)

    made = [n for n in os.listdir(dist)] if os.path.isdir(dist) else []
    print("\n완료 →", dist)
    for n in made:
        p = os.path.join(dist, n)
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) \
            else os.path.getsize(p)
        print(f"  {n}  ({sz/1e6:.0f} MB)")
    if not os.path.isdir(VENDOR):
        print("\nffmpeg 미포함 빌드입니다 — 앱이 PATH 에서 찾고, 없으면\n"
              "첫 실행 화면이 설치 방법을 안내합니다.")


if __name__ == "__main__":
    main()
