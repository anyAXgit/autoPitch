#!/usr/bin/env python3
"""autoPitch launcher — the entry point for the packaged app.

Double-clicking the app lands here: pick a writable workspace, start the local
server, open the browser. Nothing is rendered or configured yet; the setup
screen in the browser handles the rest (and reports a missing ffmpeg there
rather than dying in a terminal the user never sees).

Also usable from a checkout:  python autopitch.py [--root DIR] [--port N]
"""
import argparse
import os
import shutil
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_PORT = 8756


def _frozen():
    return getattr(sys, "frozen", False)


def default_workspace():
    """Where a packaged app keeps the user's project.

    A .app bundle's own directory is read-only (and on macOS may be quarantined),
    so a frozen build never writes next to itself -- it uses ~/autoPitch. A
    source checkout keeps working in place, which is what developers expect.
    """
    if not _frozen():
        return HERE
    return os.path.join(os.path.expanduser("~"), "autoPitch")


def ensure_workspace(root):
    """Create the folder layout and seed config.yaml on first run."""
    for sub in ("data/raw/cam1", "data/raw/cam2", "data/output", "data/_gui",
                "data/temp_audio", "data/temp_video"):
        os.makedirs(os.path.join(root, *sub.split("/")), exist_ok=True)
    cfg = os.path.join(root, "config.yaml")
    if not os.path.exists(cfg):
        seed = os.path.join(_resource_dir(), "config.yaml")
        if os.path.exists(seed):
            shutil.copyfile(seed, cfg)
    return root


def _resource_dir():
    """Read-only files that ship with the build (config template, gui/)."""
    return getattr(sys, "_MEIPASS", HERE)


def _free_port(preferred):
    import socket
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("사용 가능한 포트를 찾지 못했습니다.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="autoPitch")
    ap.add_argument("--root", default=None, help="작업 폴더 (기본: 홈의 autoPitch)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)

    root = os.path.realpath(os.path.expanduser(a.root or default_workspace()))
    ensure_workspace(root)

    from gui import server as gui_server
    from http.server import ThreadingHTTPServer

    gui_server.STATE["root"] = root
    port = _free_port(a.port)
    url = f"http://127.0.0.1:{port}"

    httpd = ThreadingHTTPServer(("127.0.0.1", port), gui_server.Handler)
    print(f"autoPitch — {url}\n작업 폴더: {root}\n종료하려면 이 창을 닫으세요.")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
