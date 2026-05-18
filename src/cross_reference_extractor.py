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

from .config import CONFIDENCE_REGEX
from .law_id_resolver import resolve as resolve_law_id

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
# Patterns cover both verbal forms (modifică, completează) and noun+definite-article
# forms (modificarea, completarea) common in Romanian legal preambles.
_VERB_TRIGGERS = [
    (re.compile(r"(?i)\b(modific(?:ă|area|are|at|ată))\b"), "modifică"),
    (re.compile(r"(?i)\b(complet(?:ează|area|are|at|ată))\b"), "completează"),
    (re.compile(r"(?i)\b(abrog(?:ă|area|are|at|ată))\b"), "abroga"),
    (re.compile(r"(?i)\b(introduc(?:erea|ere|e)?)\b"), "introduce"),
    (re.compile(r"(?i)\b(republic(?:ă|area|are|at|ată))\b"), "republică"),
    (re.compile(r"(?i)\b(aprob(?:ă|area|are|at|ată))\b"), "aprobă"),
    (re.compile(r"(?i)\btranspun(?:e|erea|ere|s|să)\b"), "transpune"),
]

# Shared stem alternatives used to build the compound-trigger pattern.
_TRIGGER_NOUN_OR_VERB = (
    r"modific(?:ă|area|are|at|ată)"
    r"|complet(?:ează|area|are|at|ată)"
    r"|abrog(?:ă|area|are|at|ată)"
    r"|introduc(?:erea|ere|e)?"
    r"|republic(?:ă|area|are|at|ată)"
    r"|aprob(?:ă|area|are|at|ată)"
    r"|transpun(?:e|erea|ere|s|să)"
)

# Detects conjunctive pairs like "modificarea și completarea" or
# "modifică și completează", capturing each verb form as a group.
_COMPOUND_TRIGGER = re.compile(
    r"(?i)\b(" + _TRIGGER_NOUN_OR_VERB + r")\b\s+(?:și|sau)\s+\b(" + _TRIGGER_NOUN_OR_VERB + r")\b"
)


def _token_to_canonical(token: str) -> str | None:
    for pattern, canonical in _VERB_TRIGGERS:
        if pattern.search(token):
            return canonical
    return None


# Maximum chars between a trigger verb and the act citation for them to be
# considered related. Keeps us from linking distant unrelated mentions.
_TRIGGER_WINDOW = 80


def _best_trigger(window: str) -> tuple[int, str | None]:
    """Return (position-of-rightmost-trigger, canonical) within `window`.

    Position is the trigger's start index inside the window; -1 if not found.
    """
    best_pos = -1
    best_rel: str | None = None
    for pattern, canonical in _VERB_TRIGGERS:
        for m in pattern.finditer(window):
            if m.start() > best_pos:
                best_pos = m.start()
                best_rel = canonical
    return best_pos, best_rel


def _find_relation_for_span(text: str, span_start: int, span_end: int) -> list[str]:
    """Return the list of canonical relations for the citation at [span_start, span_end).

    Checks for a conjunctive compound expression (e.g. "modificarea și completarea")
    first and returns both relations when found. Falls back to the single closest
    trigger verb otherwise.
    """
    window_lo = max(0, span_start - _TRIGGER_WINDOW)
    window_hi = min(len(text), span_end + _TRIGGER_WINDOW)
    compound = _COMPOUND_TRIGGER.search(text[window_lo:window_hi])
    if compound:
        rel1 = _token_to_canonical(compound.group(1))
        rel2 = _token_to_canonical(compound.group(2))
        rels = list(dict.fromkeys(r for r in (rel1, rel2) if r))
        if rels:
            return rels

    # Left window: text[L:span_start] -> rightmost trigger is closest.
    left_window = text[window_lo:span_start]
    left_pos, left_rel = _best_trigger(left_window)
    left_dist = (len(left_window) - left_pos) if left_pos >= 0 else None

    # Right window: text[span_end:R]. We want the LEFTMOST trigger here (closest
    # to the citation), and only if no other act citation appears before it.
    right_window = text[span_end:window_hi]
    right_pos, right_rel = -1, None
    for pattern, canonical in _VERB_TRIGGERS:
        m = pattern.search(right_window)
        if m and (right_pos == -1 or m.start() < right_pos):
            right_pos, right_rel = m.start(), canonical
    if right_pos >= 0:
        next_act = _ACT_PATTERN.search(right_window)
        if next_act and next_act.start() < right_pos:
            right_pos, right_rel = -1, None
    right_dist = right_pos if right_pos >= 0 else None

    # Pick the closer side; left wins on ties.
    if left_dist is None and right_dist is None:
        return []
    if right_dist is None:
        return [left_rel] if left_rel else []
    if left_dist is None:
        return [right_rel] if right_rel else []
    rel = left_rel if left_dist <= right_dist else right_rel
    return [rel] if rel else []


# Maps law_id tip prefix -> set of tokens that may appear in a citation.
_TIP_TOKENS: dict[str, tuple[str, ...]] = {
    "lege": ("leg",),  # Legea, Legii, Lege-cadru
    "oug": ("o.u.g", "oug", "ordonanț"),  # Ordonanța de urgență / O.U.G.
    "og": ("o.g", "ordonanț"),
    "hg": ("h.g", "hotărâr"),
    "decret": ("decret",),
}


def _is_self_citation(law_id: str, cited: str) -> bool:
    """Return True if `cited` refers to the law identified by `law_id`."""
    if not law_id:
        return False
    parts = law_id.split("_")
    if len(parts) < 3:
        return False
    tip, numar, an = parts[0].lower(), parts[-2], parts[-1]

    cited_low = cited.lower()
    cited_num = re.sub(r"[.\s]", "", cited_low)
    if f"{numar}/{an}" not in cited_num:
        return False

    expected = _TIP_TOKENS.get(tip, (tip,))
    return any(tok in cited_low for tok in expected)


def extract_cross_references(text: str, current_law_id: str) -> pd.DataFrame:
    """Extract triples for all cited acts in ``text``."""
    triples: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()

    for m in _ACT_PATTERN.finditer(text):
        surface = re.sub(r"\s+", " ", m.group(0)).strip()
        cited = resolve_law_id(surface) or surface
        if cited == current_law_id or _is_self_citation(current_law_id, surface):
            continue

        relations = _find_relation_for_span(text, m.start(), m.end()) or ["face_referire_la"]
        for relation in relations:
            triple = (current_law_id, relation, cited)
            if triple in seen:
                continue
            seen.add(triple)
            triples.append(list(triple))

    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    if not df.empty:
        df["source_method"] = "regex"
        df["confidence"] = CONFIDENCE_REGEX
    else:
        df["source_method"] = pd.Series(dtype="object")
        df["confidence"] = pd.Series(dtype="float64")
    return df


if __name__ == "__main__":
    sample = (
        "Articolul II Ordonanța de urgență a Guvernului nr. 89/2025 pentru "
        "modificarea și completarea Legii nr. 227/2015 privind Codul fiscal, "
        "publicată în Monitorul Oficial al României, Partea I, nr. 1203 din "
        "24 decembrie 2025, se modifică și se completează după cum urmează: "
        "se abrogă art. 3 din Hotărârea Guvernului nr. 1.705/2006."
    )
    print(extract_cross_references(sample, current_law_id="oug_13_2026").to_string())
