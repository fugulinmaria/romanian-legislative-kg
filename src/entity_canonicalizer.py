"""Conservative canonicalizer for entity strings in extracted triples.

Applies cosmetic normalization (NFC, diacritics, whitespace, punctuation,
casing of bare law-type words) so that surface variants of the same entity
collapse to a single graph node. Numbered citations (`Legea nr. 53/2003`)
and `law_id` values (`lege_53_2003`) are preserved verbatim apart from
whitespace/diacritic cleanup.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Cedilla -> comma-below diacritic map (mirrors text_normalizer).
_DIACRITIC_MAP = str.maketrans(
    {
        "\u015f": "\u0219",
        "\u015e": "\u0218",  # ş/Ş -> ș/Ș
        "\u0163": "\u021b",
        "\u0162": "\u021a",  # ţ/Ţ -> ț/Ț
    }
)

# Zero-width and NBSP characters to strip.
_ZW_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00a0]")
_WS_RE = re.compile(r"\s+")
_TRIM_PUNCT = " \t\n\r.,;:\"'`«»“”’()[]{}"

# Bare law-type words that should be title-cased when they appear alone or
# at the start of a non-numbered cell.
_LAW_WORDS = {
    "legea": "Legea",
    "codul": "Codul",
    "hotărârea": "Hotărârea",
    "ordonanța": "Ordonanța",
    "decretul": "Decretul",
    "constituția": "Constituția",
    "monitorul": "Monitorul",
    "parlamentul": "Parlamentul",
    "guvernul": "Guvernul",
    "președintele": "Președintele",
    "ministerul": "Ministerul",
}

_LAW_ID_RE = re.compile(r"^[a-z]+(?:_\d+){2,}$")  # e.g. lege_53_2003


def canonicalize(value: str) -> str:
    """Return a canonical form of an entity string."""
    if value is None:
        return value
    s = str(value)

    # 1. Unicode + diacritics
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_DIACRITIC_MAP)

    # 2. Whitespace cleanup
    s = _ZW_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip(_TRIM_PUNCT)

    if not s:
        return s

    # 3. Preserve law_id verbatim
    if _LAW_ID_RE.match(s):
        return s

    # 4. Numbered citations: normalize spacing around 'nr.' / '/' but keep case
    #    of the head word (Legea / LEGEA both become 'Legea').
    has_number = bool(re.search(r"\d", s))
    if has_number:
        s = re.sub(r"\bnr\s*\.\s*", "nr. ", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*/\s*", "/", s)
        first_word, sep, rest = s.partition(" ")
        lw = first_word.lower().rstrip(".,")
        if lw in _LAW_WORDS:
            s = _LAW_WORDS[lw] + sep + rest
        return s

    # 5. Bare phrase: title-case the first known law-type word; leave rest alone.
    first_word, sep, rest = s.partition(" ")
    lw = first_word.lower()
    if lw in _LAW_WORDS:
        s = _LAW_WORDS[lw] + sep + rest

    return s


def canonicalize_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize the `head` and `tail` columns of a triples DataFrame."""
    if df.empty:
        return df
    out = df.copy()
    for col in ("head", "tail"):
        if col in out.columns:
            out[col] = out[col].map(canonicalize)
    return out
