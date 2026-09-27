"""Console helpers.

Windows consoles often use a legacy code page (cp1252, cp437, ...) that cannot
print every Unicode character. A single unprintable character in a document
would then crash the CLI with UnicodeEncodeError. The shipped demo documents
and all messages are plain ASCII, but your own documents (especially PDFs)
may not be, so the CLIs call make_console_safe() once at start-up.
"""

from __future__ import annotations

import sys


def make_console_safe() -> None:
    """Replace unprintable characters with '?' instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
