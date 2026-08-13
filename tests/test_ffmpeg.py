"""Finding the binaries when the app was not launched from a shell.

An app opened from Finder inherits `/usr/bin:/bin:/usr/sbin:/sbin` and nothing
else. Homebrew lives outside that, so a user who has ffmpeg -- and brew, and can
run both in a terminal -- was told by the app to install what they already had.

Kept free of POSIX assumptions: the separator is `;` on Windows, and there a
file is only executable if its extension says so, so a bare `brew` is invisible
to `shutil.which` no matter which directory it sits in.
"""
import os

import pytest

from src import ffmpeg as F

EXE_SUFFIX = ".exe" if os.name == "nt" else ""
SYSTEM_DIRS = [r"C:\Windows\system32"] if os.name == "nt" else ["/usr/bin", "/bin"]


def _install(directory, name):
    """A file `which` will accept as runnable, and the path it should report."""
    p = directory / (name + EXE_SUFFIX)
    p.write_text("")
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def finder_launch(monkeypatch, tmp_path):
    """A stripped PATH, plus a stand-in for /opt/homebrew/bin."""
    brewish = tmp_path / "homebrew" / "bin"
    brewish.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join(SYSTEM_DIRS))
    monkeypatch.setattr(F, "_EXTRA_BIN_DIRS", [str(brewish)])
    return brewish


def test_finds_a_binary_the_shell_path_does_not_reach(finder_launch):
    expected = _install(finder_launch, "brew")
    assert F.which("brew") == expected


def test_still_missing_is_still_missing(finder_launch):
    assert F.which("definitely-not-installed") is None


def test_path_wins_over_the_fallback(finder_launch, monkeypatch, tmp_path):
    """The fallback is a widening, not an override -- a binary the user put on
    their own PATH is the one they meant."""
    on_path = tmp_path / "mine"
    on_path.mkdir()
    expected = _install(on_path, "ffmpeg")
    _install(finder_launch, "ffmpeg")
    monkeypatch.setenv("PATH", str(on_path))
    assert F.which("ffmpeg") == expected


def test_search_path_keeps_the_original_entries(finder_launch):
    parts = F.search_path().split(os.pathsep)
    for d in SYSTEM_DIRS:
        assert d in parts
    assert str(finder_launch) in parts


def test_search_path_skips_directories_that_are_not_there(monkeypatch):
    absent = os.path.join(os.sep, "nope", "does", "not", "exist")
    monkeypatch.setattr(F, "_EXTRA_BIN_DIRS", [absent])
    assert absent not in F.search_path().split(os.pathsep)


def test_no_duplicate_entries(monkeypatch, finder_launch):
    """/usr/local/bin is on some machines' PATH already; adding it twice would
    make the value grow every time this is called."""
    monkeypatch.setenv("PATH", os.pathsep.join([SYSTEM_DIRS[0], str(finder_launch)]))
    parts = F.search_path().split(os.pathsep)
    assert parts.count(str(finder_launch)) == 1
