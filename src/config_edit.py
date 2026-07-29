"""Read and write individual config.yaml values without losing the file.

Round-tripping through PyYAML would drop every comment, and config.yaml carries
the measured reasoning behind several settings (why the quiet-goal scan is off
for this venue, why the ROI search window is biased before the onset). Losing
that to a checkbox toggle in the UI is not an acceptable trade, and pulling in
ruamel.yaml just for round-tripping is a dependency the packaged build does not
need.

So this edits the specific line in place: same indentation, same trailing
comment, only the scalar changes. Keys are dotted paths (`peak.threshold_k`)
resolved by indentation, which is enough for this file's plain nested-mapping
shape. Anything it cannot locate is reported rather than appended blindly.
"""
import re

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w-]*)\s*:(?P<rest>.*)$")


def _split_comment(rest):
    """Split ` value<gap># comment` -- naive, but this file has no '#' in values.

    The gap is returned so an edit keeps the comment in its original column;
    collapsing it to one space reflows the file and makes the diff look larger
    than the change actually is.
    """
    i = rest.find("#")
    if i < 0:
        return rest.rstrip(), "", ""
    head = rest[:i]
    value = head.rstrip()
    return value, head[len(value):], rest[i:]


def _index(lines):
    """Map dotted path -> line number, using indentation for nesting."""
    out, stack = {}, []          # stack of (indent, key)
    for n, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY.match(line.rstrip("\n"))
        if not m:
            continue
        indent = len(m["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [m["key"]])
        out[path] = n
        value, _, _ = _split_comment(m["rest"])
        if value.strip() == "":          # a mapping header, e.g. `peak:`
            stack.append((indent, m["key"]))
    return out


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        # keep 2.0 as 2.0 so the file stays obviously numeric
        return repr(round(value, 6))
    return str(value)


def _parse(text):
    t = text.strip()
    if t in ("true", "false"):
        return t == "true"
    if t in ("null", "~", ""):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t.strip("'\"")


def read_values(path, keys):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    idx = _index(lines)
    out = {}
    for k in keys:
        if k not in idx:
            continue
        m = _KEY.match(lines[idx[k]].rstrip("\n"))
        value, _, _ = _split_comment(m["rest"])
        out[k] = _parse(value)
    return out


def set_values(path, updates):
    """Apply {dotted_key: value}. Returns (changed_keys, missing_keys)."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    idx = _index(lines)
    changed, missing = [], []
    for key, value in updates.items():
        if key not in idx:
            missing.append(key)
            continue
        n = idx[key]
        m = _KEY.match(lines[n].rstrip("\n"))
        old, gap, comment = _split_comment(m["rest"])
        if _parse(old) == value:
            continue
        new = _fmt(value)
        # keep the comment where it was: shrink/grow the gap by the length delta
        pad = " " * max(1, len(gap) + len(old.strip()) - len(new)) if comment else ""
        lines[n] = f"{m['indent']}{m['key']}: {new}{pad}{comment}\n"
        changed.append(key)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed, missing
