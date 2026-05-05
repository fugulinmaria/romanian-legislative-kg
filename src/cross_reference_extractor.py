"""
Deterministic regex-based extractor for Romanian legislative cross-references.

Produces high-quality triples for citations that are unambiguous and don't need
an LLM:
  - "modifică <act>"          -> (current_law, modifică, <act>)
  - "completează <act>"       -> (current_law, completează, <act>)
  - "abrogă <act>"            -> (current_law, abroga, <act>)
  - "transpune <directiva>"   -> (current_law, transpune, <directiva>)
  - bare references           -> (current_law, face_referire_la, <act>)

Designed to run BEFORE the LLM extractor; results are merged + de-duplicated.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Citation patterns (Romanian legal forms, after diacritic normalization)
# ---------------------------------------------------------------------------

# Matches act citations like:
#   Legea nr. 53/2003
#   Legea-cadru nr. 153/2017
#   Ordonanța de urgență a Guvernului nr. 89/2025
#   O.U.G. nr. 89/2025
#   Hotărârea Guvernului nr. 1.705/2006
#   H.G. nr. 1705/2006
#   Ordonanța Guvernului nr. 81/2003
#   Codul fiscal / Codul muncii / Codul administrativ / Codul civil / Codul penal
#   Constituția României
#
# We keep the full matched span as the cited entity (preserves "nr. X/YYYY").
_ACT_PATTERN = re.compile(
    r"""(?xi)
    (?:
        Leg(?:ea|ii)(?:-cadru)?\s+nr\.?\s*\d[\d\.]*/\d{4}
      | Ordonanț(?:a|ei)\s+de\s+urgenț(?:ă|ei)\s+(?:a\s+Guvernului\s+)?nr\.?\s*\d[\d\.]*/\d{4}
      | O\.?\s*U\.?\s*G\.?\s*nr\.?\s*\d[\d\.]*/\d{4}
      | Ordonanț(?:a|ei)\s+(?:Guvernului\s+)?nr\.?\s*\d[\d\.]*/\d{4}
      | O\.?\s*G\.?\s*nr\.?\s*\d[\d\.]*/\d{4}
      | Hotărâr(?:ea|ii)\s+(?:Guvernului\s+)?nr\.?\s*\d[\d\.]*/\d{4}
      | H\.?\s*G\.?\s*nr\.?\s*\d[\d\.]*/\d{4}
      | Codul\s+(?:fiscal|muncii|administrativ|civil|penal
                 |de\s+procedur(?:ă|ei)\s+civil(?:ă|ei)
                 |de\s+procedur(?:ă|ei)\s+penal(?:ă|ei))
      | Constituți(?:a|ei)\s+României
    )
    """
)

# Trigger verbs that introduce relations to a cited act. The trigger group is
# captured so we know which canonical relation to emit.
_VERB_TRIGGERS = [
    (re.compile(r"(?i)\b(modific(?:ă|are|at|ată))\b"), "modifică"),
    (re.compile(r"(?i)\b(completeaz(?:ă|are|at|ată))\b"), "completează"),
    (re.compile(r"(?i)\b(abrog(?:ă|are|at|ată))\b"), "abroga"),
    (re.compile(r"(?i)\b(introduce|introduc(?:ere|e)?)\b"), "introduce"),
    (re.compile(r"(?i)\b(republic(?:ă|at|ată|are))\b"), "republică"),
    (re.compile(r"(?i)\b(aprob(?:ă|at|ată|are))\b"), "aprobă"),
    (re.compile(r"(?i)\btranspun(?:e|ere|s|să)\b"), "transpune"),
]

# Maximum chars between a trigger verb and the act citation for them to be
# considered related. Keeps us from linking distant unrelated mentions.
_TRIGGER_WINDOW = 80


def _find_relation_for_span(text: str, span_start: int) -> str | None:
    """Find the closest preceding trigger verb within the window."""
    window_start = max(0, span_start - _TRIGGER_WINDOW)
    window = text[window_start:span_start]
    best_rel: str | None = None
    best_pos = -1
    for pattern, canonical in _VERB_TRIGGERS:
        for m in pattern.finditer(window):
            if m.start() > best_pos:
                best_pos = m.start()
                best_rel = canonical
    return best_rel


def extract_cross_references(text: str, current_law_id: str) -> pd.DataFrame:
    """Extract triples for all cited acts in ``text``.

    Args:
        text: Article body (already normalized).
        current_law_id: ID of the law the text belongs to (used as triple head).

    Returns:
        DataFrame with columns ``head``, ``relation``, ``tail``. May be empty.
    """
    triples: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()

    for m in _ACT_PATTERN.finditer(text):
        cited = re.sub(r"\s+", " ", m.group(0)).strip()
        # Skip self-references (a law citing itself)
        if current_law_id and current_law_id.replace("_", " ") in cited.lower():
            continue

        relation = _find_relation_for_span(text, m.start()) or "face_referire_la"
        triple = (current_law_id, relation, cited)
        if triple in seen:
            continue
        seen.add(triple)
        triples.append(list(triple))

    return pd.DataFrame(triples, columns=["head", "relation", "tail"])


if __name__ == "__main__":
    sample = (
        "Articolul II Ordonanța de urgență a Guvernului nr. 89/2025 pentru "
        "modificarea și completarea Legii nr. 227/2015 privind Codul fiscal, "
        "publicată în Monitorul Oficial al României, Partea I, nr. 1203 din "
        "24 decembrie 2025, se modifică și se completează după cum urmează: "
        "se abrogă art. 3 din Hotărârea Guvernului nr. 1.705/2006."
    )
    print(extract_cross_references(sample, current_law_id="oug_13_2026").to_string())
