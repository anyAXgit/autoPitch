"""The one exception type whose text is written for the person running the app."""


class UserError(Exception):
    """A message written for the user. Shown as-is, without the class name --
    "FileNotFoundError: 원본 영상을..." reads like a crash; the sentence alone
    reads like an instruction.

    Raise this for a situation the user can fix (a missing file, an optional
    package that isn't installed, a setting that needs changing). Anything else
    should keep its own type, so a real bug still looks like a real bug.
    """
