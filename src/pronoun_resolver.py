"""Resolve anaphoric references in extracted triples to the current law_id.

LLM and regex extractors frequently surface heads/tails like "prezenta lege",
"prezentul cod fiscal" or a bare "Hotărârea". These all refer to the document
the article belongs to and must be rewritten to the canonical `law_id` so
deduplication and graph queries work across articles.
"""

from __future__ import annotations

import re

import pandas as pd

# Anchored-at-start pronoun pattern. We allow optional trailing words
# (e.g. "prezenta lege organică", "prezentul cod fiscal") so the entire cell
# collapses to law_id.
_PRONOUN_RE = re.compile(
    r"^(?:"
    r"prezent[au]l?\b.*"  # prezenta / prezentul / prezentul cod ...
    r"|hot[ăa]r[âa]rea\b.*"  # Hotărârea (Guvernului)?
    r"|legea\b.*"  # Legea (de față)?
    r"|ordonan[țt]a\b.*"  # Ordonanța / Ordonanța de urgență
    r"|codul\b.*"  # Codul (muncii|fiscal|...)
    r")$",
    re.IGNORECASE,
)


def resolve_pronouns(df: pd.DataFrame, law_id: str) -> pd.DataFrame:
    """Rewrite anaphoric heads/tails in `df` to `law_id`.

    Returns a new DataFrame; `df` is not mutated.
    """
    if df.empty or not law_id:
        return df

    out = df.copy()
    for col in ("head", "tail"):
        if col not in out.columns:
            continue
        cells = out[col].astype(str)
        is_explicit = cells.str.contains(r"\d|\bnr\.", case=False, na=False)
        mask = cells.str.match(_PRONOUN_RE, na=False) & ~is_explicit
        out.loc[mask, col] = law_id
    return out
