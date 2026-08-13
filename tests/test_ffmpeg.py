"""Finding the binaries when the app was not launched from a shell.

An app opened from Finder inherits `/usr/bin:/bin:/usr/sbin:/sbin` and nothing
else. Homebrew lives outside that, so a user who has ffmpeg -- and brew, and can
run both in a terminal -- was told by the app to install what they already had.
"""
import os

import pytest

from src import ffmpeg as F


@pytest.fixture
def finder_launch(monkeypatch, tmp_path):
    """A stripped PATH, plus a stand-in for /opt/homebrew/bin."""
    brewish = tmp_path / "homebrew" / "bin"
    brewish.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(F, "_EXTRA_BIN_DIRS", [str(brewish)])
    return brewish


def _install(d, name):
    p = d / name
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


def test_finds_a_binary_the_shell_path_does_not_reach(finder_launch):
    _install(finder_launch, "brew")
    assert F.which("brew") == str(finder_launch / "brew")


def test_still_missing_is_still_missing(finder_launch):
    assert F.which("definitely-not-installed") is None


def test_path_wins_over_the_fallback(finder_launch, monkeypatch, tmp_path):
    """The fallback is a widening, not an override -- a binary the user put on
    their own PATH is the one they meant."""
    on_path = tmp_path / "mine"
    on_path.mkdir()
    _install(on_path, "ffmpeg")
    _install(finder_launch, "ffmpeg")
    monkeypatch.setenv("PATH", str(on_path))
    assert F.which("ffmpeg") == str(on_path / "ffmpeg")


def test_search_path_keeps_the_original_entries(finder_launch):
    parts = F.search_path().split(os.pathsep)
    assert "/usr/bin" in parts and "/bin" in parts
    assert str(finder_launch) in parts


def test_search_path_skips_directories_that_are_not_there(monkeypatch):
    monkeypatch.setattr(F, "_EXTRA_BIN_DIRS", ["/nope/does/not/exist"])
    assert "/nope/does/not/exist" not in F.search_path()


def test_no_duplicate_entries(monkeypatch, finder_launch):
    """/usr/local/bin is on some machines' PATH already; adding it twice would
    make the value grow every time this is called."""
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}{finder_launch}")
    parts = F.search_path().split(os.pathsep)
    assert parts.count(str(finder_launch)) == 1
