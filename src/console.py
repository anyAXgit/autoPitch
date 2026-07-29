"""Make stdout/stderr safe for the Korean text this project prints.

A Windows console is cp949 (Korean locale) or cp1252, and printing Hangul or an
em-dash to it raises UnicodeEncodeError -- which killed the packaged app on its
very first line of output. Entry points call `enable_utf8()` before printing
anything.

`errors="replace"` rather than "strict": a console that genuinely cannot render
a character should show a placeholder, not take the whole run down with it.
"""
import sys


def enable_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass            # already detached, or a stream without reconfigure
