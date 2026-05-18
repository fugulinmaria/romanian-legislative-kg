"""Resolve surface-form legislative citations to canonical ``law_id``.

Builds an in-memory alias index from law metadata and exposes a single
``resolve()`` lookup. Unknown but well-formed citations (``Legea nr. N/YYYY``)
are synthesized into a deterministic ``law_id`` so external acts still collapse
to one node across documents.
"""

from __future__ import annotations

import re
import unicodedata

from .text_normalizer import _DIACRITIC_MAP

# tip_act (free-form, accented) -> law_id prefix
_TIP_PREFIX: dict[str, str] = {
    "lege": "lege",
    "legea": "lege",
    "lege-cadru": "lege",
    "ordonanta de urgenta": "oug",
    "ordonanta de urgenta a guvernului": "oug",
    "oug": "oug",
    "o.u.g": "oug",
    "ordonanta": "og",
    "ordonanta a guvernului": "og",
    "ordonanta guvernului": "og",
    "og": "og",
    "o.g": "og",
    "hotarare": "hg",
    "hotarare de guvern": "hg",
    "hotararea guvernului": "hg",
    "hg": "hg",
    "h.g": "hg",
    "decret": "decret",
    "constitutia": "constitutia",
    "constitutia romaniei": "constitutia_romaniei",
}

# Code names (no number) -> stable law_id. The actual Codul muncii (lege_53_2003)
# is overridden when its meta is registered.
_CODE_LAW_IDS: dict[str, str] = {
    "codul muncii": "lege_53_2003",
    "codul fiscal": "lege_227_2015",
    "codul civil": "lege_287_2009",
    "codul penal": "lege_286_2009",
    "codul administrativ": "oug_57_2019",
    "codul de procedura civila": "lege_134_2010",
    "codul de procedura penala": "lege_135_2010",
    "constitutia romaniei": "constitutia_romaniei",
    "constitutia": "constitutia_romaniei",
}

_LAW_ID_RE = re.compile(r"^[a-z]+(?:_\d+){2,}$")
_NUMBERED_RE = re.compile(
    r"""(?xi)
    ^(?P<tip>.+?)\s*nr\.?\s*(?P<numar>\d[\d\.]*)\s*/\s*(?P<an>\d{4})\s*$
    """
)
# Same, but no "nr." (e.g. "Legea 53/2003")
_NUMBERED_NO_NR_RE = re.compile(
    r"""(?xi)
    ^(?P<tip>[A-Za-zĂÂÎȘȚăâîșț\.\- ]+?)\s+(?P<numar>\d[\d\.]*)\s*/\s*(?P<an>\d{4})\s*$
    """
)

_alias_index: dict[str, str] = {}
_law_meta: dict[str, dict] = {}


def _norm_key(s: str) -> str:
    """Aggressive normalization used for index keys and lookups."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(_DIACRITIC_MAP)
    # strip remaining diacritics
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r.,;:\"'`«»“”’()[]{}")
    return s


def _tip_to_prefix(tip_raw: str) -> str | None:
    key = _norm_key(tip_raw).rstrip(".")
    if key in _TIP_PREFIX:
        return _TIP_PREFIX[key]
    # Try first token (handles "Hotarare guvern", "Lege organica" etc.)
    head = key.split(" ", 1)[0]
    return _TIP_PREFIX.get(head)


def _clean_numar(numar: str) -> str:
    return re.sub(r"\.", "", numar)


def _canonical_citations(tip: str, numar: str, an: str) -> list[str]:
    """Generate human-readable forms registered as aliases for the law."""
    n = _clean_numar(numar)
    forms: list[str] = []
    tip_key = _norm_key(tip).rstrip(".")
    prefix = _TIP_PREFIX.get(tip_key) or _TIP_PREFIX.get(tip_key.split(" ", 1)[0])
    if prefix == "lege":
        forms += [f"legea nr. {n}/{an}", f"legea {n}/{an}", f"lege nr. {n}/{an}"]
    elif prefix == "oug":
        forms += [
            f"ordonanta de urgenta a guvernului nr. {n}/{an}",
            f"ordonanta de urgenta nr. {n}/{an}",
            f"oug nr. {n}/{an}",
            f"o.u.g. nr. {n}/{an}",
            f"o.u.g. {n}/{an}",
        ]
    elif prefix == "og":
        forms += [
            f"ordonanta guvernului nr. {n}/{an}",
            f"ordonanta nr. {n}/{an}",
            f"og nr. {n}/{an}",
            f"o.g. nr. {n}/{an}",
        ]
    elif prefix == "hg":
        forms += [
            f"hotararea guvernului nr. {n}/{an}",
            f"hotararea nr. {n}/{an}",
            f"hg nr. {n}/{an}",
            f"h.g. nr. {n}/{an}",
            f"h.g. {n}/{an}",
        ]
    elif prefix == "decret":
        forms += [f"decretul nr. {n}/{an}", f"decret nr. {n}/{an}"]
    return forms


def register_law(meta: dict) -> None:
    """Index a law's metadata so future ``resolve()`` calls find its law_id."""
    law_id = meta.get("law_id")
    if not law_id:
        return
    _law_meta[law_id] = meta
    _alias_index[_norm_key(law_id)] = law_id

    tip = meta.get("tip_act", "")
    numar = meta.get("numar", "")
    an = meta.get("an", "")
    if tip and numar and an:
        for form in _canonical_citations(tip, str(numar), str(an)):
            _alias_index[_norm_key(form)] = law_id

    for alias in meta.get("aliases", []) or []:
        _alias_index[_norm_key(alias)] = law_id


def _synthesize_law_id(citation: str) -> str | None:
    """Return a synthetic law_id for a well-formed but unknown citation."""
    key = _norm_key(citation)
    m = _NUMBERED_RE.match(key) or _NUMBERED_NO_NR_RE.match(key)
    if not m:
        return None
    prefix = _tip_to_prefix(m.group("tip"))
    if not prefix:
        return None
    numar = _clean_numar(m.group("numar"))
    an = m.group("an")
    return f"{prefix}_{numar}_{an}"


def resolve(citation: str) -> str | None:
    """Return canonical law_id for ``citation`` (registered, code name, or synthesized)."""
    if citation is None:
        return None
    s = str(citation).strip()
    if not s:
        return None
    if _LAW_ID_RE.match(s):
        return s
    key = _norm_key(s)
    if not key:
        return None
    if key in _alias_index:
        return _alias_index[key]
    if key in _CODE_LAW_IDS:
        lid = _CODE_LAW_IDS[key]
        _alias_index[key] = lid
        return lid
    synth = _synthesize_law_id(s)
    if synth:
        _alias_index[key] = synth
        return synth
    return None


def known_law_ids() -> set[str]:
    return set(_law_meta.keys())


def get_meta(law_id: str) -> dict | None:
    return _law_meta.get(law_id)


def _reset_for_tests() -> None:
    _alias_index.clear()
    _law_meta.clear()
