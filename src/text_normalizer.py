"""
Romanian legislative text normalizer.

Cleans raw .txt files from legislatie.just.ro so they are ready for embedding,
chunking and triple extraction:

  1. Unicode NFC normalization
  2. Cedilla -> comma-below diacritics (ş/ţ -> ș/ț)
  3. Strip the metadata header block (already captured in .meta.json)
  4. Normalize whitespace, preserve paragraph structure

Usage (CLI):
    python3 -m src.text_normalizer data/raw_laws/lege_24_2000.txt
"""

import re
import sys
import unicodedata
from pathlib import Path

_DIACRITIC_MAP = str.maketrans(
    {
        "\u015f": "\u0219",  # ş -> ș
        "\u015e": "\u0218",  # Ş -> Ș
        "\u0163": "\u021b",  # ţ -> ț
        "\u0162": "\u021a",  # Ţ -> Ț
    }
)

# Matches the "Publicat în  MONITORUL OFICIAL nr. X din ... YYYY" line that
# closes every legislatie.just.ro header block.
_HEADER_END_RE = re.compile(
    r"^.*MONITORUL\s+OFICIAL\s+nr\.?\s*\d+\s+din\s+\d{1,2}\s+\S+\s+\d{4}.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Three or more consecutive newlines -> two
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Trailing spaces/tabs at end of any line
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _strip_header(text: str) -> str:
    """Remove the legislatie.just.ro header block.

    The header always ends with a 'Publicat în MONITORUL OFICIAL nr. X din ...'
    line. We drop everything up to and including that line. If no such line is
    found, the text is returned unchanged.
    """
    match = None
    for m in _HEADER_END_RE.finditer(text):
        match = m  # take the last match in case "MONITORUL OFICIAL" reappears
        # in body cross-references; the header one is always near the top, so
        # finditer + last would be wrong. Use the FIRST match instead:
        break
    if match is None:
        return text
    return text[match.end() :].lstrip("\n")


def normalize_ro(text: str) -> str:
    """Normalize Romanian legislative text for downstream processing."""
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Standardize diacritics
    text = text.translate(_DIACRITIC_MAP)
    # 3. Strip header block (info already in .meta.json)
    text = _strip_header(text)
    # 4. Whitespace cleanup
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 -m src.text_normalizer <path-to-txt>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    cleaned = normalize_ro(raw)

    print("=" * 70)
    print(f"BEFORE  ({len(raw)} chars, first 400)")
    print("=" * 70)
    print(raw[:400])
    print()
    print("=" * 70)
    print(f"AFTER   ({len(cleaned)} chars, first 400)")
    print("=" * 70)
    print(cleaned[:400])
