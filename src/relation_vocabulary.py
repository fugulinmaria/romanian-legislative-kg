"""
Canonical Romanian legislative relation vocabulary.

Centralizes the allowed relation set so the prompt, the parser, the regex
pre-pass and the ontology reasoner all agree on one closed list. Synonyms
produced by the LLM are normalized to the canonical form.
"""

from __future__ import annotations

# Canonical relation -> short Romanian description (used in the prompt).
ALLOWED_RELATIONS: dict[str, str] = {
    "emis_de": "cine a emis actul (Parlamentul, Guvernul, Președintele)",
    "promulgat_de": "cine a promulgat actul (Președintele)",
    "publicat_în": "unde a fost publicat (Monitorul Oficial)",
    "modifică": "ce act/articol modifică",
    "completează": "ce act/articol completează",
    "abroga": "ce act/articol abrogă",
    "introduce": "ce articol nou introduce",
    "republică": "ce act republică",
    "aprobă": "ce act aprobă",
    "face_referire_la": "trimitere către alt act sau articol (citation)",
    "intră_în_vigoare": "data sau condiția de intrare în vigoare",
    "are_sediul_în": "unde are sediul o entitate",
    "responsabil_pentru": "cine este responsabil pentru o sarcină",
    "colaborează_cu": "cine colaborează cu cine",
    "se_aplică": "domeniul/categoria căreia i se aplică o normă",
    "transpune": "ce directivă/regulament UE transpune",
}

# Synonym -> canonical (lower-cased lookup).
# Captures gender/agreement variants and common LLM rewordings.
RELATION_SYNONYMS: dict[str, str] = {
    "publicată_în": "publicat_în",
    "publicate_în": "publicat_în",
    "publicat în": "publicat_în",
    "publicat": "publicat_în",
    "emisă_de": "emis_de",
    "emise_de": "emis_de",
    "emis de": "emis_de",
    "emis": "emis_de",
    "modifică_pe": "modifică",
    "modificată": "modifică",
    "modifica": "modifică",
    "promulgată_de": "promulgat_de",
    "promulgat de": "promulgat_de",
    "abrogă": "abroga",
    "abrogată": "abroga",
    "abrogat": "abroga",
    "republicată": "republică",
    "republicat": "republică",
    "aprobată": "aprobă",
    "aprobat": "aprobă",
    "intra_în_vigoare": "intră_în_vigoare",
    "intra în vigoare": "intră_în_vigoare",
    "se_referă_la": "face_referire_la",
    "referă_la": "face_referire_la",
    "completată": "completează",
    "completată_de": "completează",
}


def normalize_relation(rel: str) -> str | None:
    """Normalize an LLM-produced relation token to a canonical relation.

    Returns:
        The canonical relation name, or ``None`` if the token does not map
        to any known relation (caller should drop the triple).
    """
    if not rel:
        return None
    key = rel.strip().lower().replace(" ", "_")
    if key in ALLOWED_RELATIONS:
        return key
    if key in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[key]
    return None


def prompt_vocabulary_block() -> str:
    """Render the relation list as a prompt fragment."""
    lines = [f"- {name}: {desc}" for name, desc in ALLOWED_RELATIONS.items()]
    return "\n".join(lines)
