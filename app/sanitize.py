"""Post-processing to normalize LLM caption output into clean plain text."""

import re

# Matches "Option 1:", "Option 1 (The "X" Vibe):", "**Option 1**", etc.
_OPTION_HEADER_RE = re.compile(
    r"^\s*(\*\*)?option\s*\d+.*?(\*\*)?\s*:?\s*$", re.IGNORECASE | re.MULTILINE
)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_MARKDOWN_ASTERISK_RE = re.compile(r"\*+")
_LEADING_LABEL_RE = re.compile(
    r"^\s*(caption|here'?s? (a|the) caption)\s*:?\s*", re.IGNORECASE
)
# All straight and curly quote characters, anywhere in the string
_INTERNAL_QUOTES_RE = re.compile(r'["\u201c\u201d\u2018\u2019]')


def sanitize_caption(text: str) -> str:
    """Clean a raw model caption into a single plain-text line.

    Strips markdown, escape characters, embedded quote marks, and multi-option
    formatting. If the model returned multiple options despite instructions,
    keeps only the first substantive one.
    """
    if not text:
        return text

    # Normalize escaped characters models sometimes emit
    cleaned = text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    cleaned = cleaned.replace("\\", "")

    # If multiple "Option N" blocks exist, keep only the first non-header line
    if _OPTION_HEADER_RE.search(cleaned):
        lines = [
            ln.strip() for ln in cleaned.splitlines()
            if ln.strip() and not _OPTION_HEADER_RE.match(ln)
        ]
        cleaned = lines[0] if lines else cleaned

    # Strip markdown bold/asterisks
    cleaned = _MARKDOWN_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_ASTERISK_RE.sub("", cleaned)

    # Strip leading labels like "Caption:" / "Here's a caption:"
    cleaned = _LEADING_LABEL_RE.sub("", cleaned)

    # Remove ALL quote characters (straight and curly), not just leading/trailing —
    # models often wrap an emphasized phrase mid-sentence, not just the whole caption
    cleaned = _INTERNAL_QUOTES_RE.sub("", cleaned)

    # Collapse whitespace/newlines into a single line
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()