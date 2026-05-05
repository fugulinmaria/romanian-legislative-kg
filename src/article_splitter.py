"""
Article-level splitter for Romanian legislative texts.

Given a normalized law body (output of ``normalize_ro``), produces one chunk per
article so each chunk can be:
  - embedded as a coherent, citable retrieval unit, and
  - sent independently to the LLM for triple extraction.

Recognized article header forms:
  - ``Articolul 1``
  - ``Art. 1``
  - ``Articolul I``           (Roman numerals, common in OUGs)
  - ``Articolul 5^1``         (renumbered articles)
  - ``ARTICOL UNIC``          (single-article HGs)

The text BEFORE the first article header is preserved as a ``preambul`` chunk
(recitals, "Având în vedere ...", "în temeiul ..." etc.) when non-trivial.

Usage (CLI):
    python3 -m src.article_splitter data/raw_laws/oug_13_2026.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches an article header at the start of a line.
# Group "num"  -> article number (Arabic, Roman, or "N^M")
# Group "unic" -> non-empty when the header is "ARTICOL UNIC"
_ARTICLE_HEADER_RE = re.compile(
    r"^[ \t]*(?:"
    r"ART(?:ICOLUL)?\.?[ \t]+(?P<num>\d+(?:\^\d+)?|[IVXLCDM]+)"
    r"|"
    r"(?P<unic>ARTICOL[ \t]+UNIC)"
    r")\b[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Minimum chars for the pre-first-article text to be kept as "preambul".
_MIN_PREAMBLE_CHARS = 40


def _normalize_number(num: str | None, unic: str | None) -> str:
    if unic:
        return "unic"
    return (num or "").strip()


_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _is_roman(num: str) -> bool:
    return bool(_ROMAN_RE.match(num))


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN_VALUES.get(ch, 0)
        total += -v if v < prev else v
        prev = v
    return total


def _base_number(num: str) -> int:
    """Return the comparable base number for sequencing.

    Examples:
        "1"    -> 1
        "14^2" -> 14
        "III"  -> 3
        "unic" -> 0 (always accepted)
    """
    if num == "unic":
        return 0
    if _is_roman(num):
        return _roman_to_int(num)
    base = num.split("^", 1)[0]
    try:
        return int(base)
    except ValueError:
        return -1


def _filter_by_dominant_style(matches: list[re.Match]) -> list[re.Match]:
    """Keep only top-level article headers.

    Romanian legislative texts use ONE numbering style at the top level:
      - OUGs / amending acts -> Roman numerals (I, II, III, ...)
      - Laws / codes         -> Arabic numerals (1, 2, 3, ...)
      - Short HGs            -> a single ARTICOL UNIC

    Article bodies frequently QUOTE inserted/modified article text from other
    acts (e.g. Article 15 of Law 190/2018 quotes ``Articolul 14^2``; Article II
    of OUG 13/2026 quotes ``Articolul V`` of another act). Editors' notes at
    the top of republished texts can also introduce stray Roman numerals
    (e.g. ``Articolul IV din LEGEA 283/2024`` at the top of Codul muncii).

    Strategy:
      1. If any ARTICOL UNIC header is present, treat the doc as single-article
         and return only that.
      2. Group matches by style (Roman vs Arabic). Pick the style with MORE
         matches as the document's dominant numbering.
      3. Within that style, return the LONGEST STRICTLY-INCREASING subsequence
         of base numbers — this naturally rejects nested quotes and stray
         out-of-order headers.
    """
    if not matches:
        return matches

    # 1. ARTICOL UNIC short-circuit
    unic_matches = [m for m in matches if m.group("unic")]
    if unic_matches:
        return unic_matches[:1]

    # 2. Pick dominant style by frequency
    arabic = [m for m in matches if not _is_roman(m.group("num") or "")]
    roman = [m for m in matches if _is_roman(m.group("num") or "")]
    candidates = arabic if len(arabic) >= len(roman) else roman

    if not candidates:
        return []

    # 3. Longest strictly-increasing subsequence by base number
    return _longest_increasing(candidates)


def _longest_increasing(matches: list[re.Match]) -> list[re.Match]:
    """Return matches forming the longest strictly-increasing subsequence
    (by base article number), preserving original document order."""
    n = len(matches)
    if n <= 1:
        return matches

    bases = [_base_number(_normalize_number(m.group("num"), m.group("unic"))) for m in matches]

    # Standard O(n^2) LIS with parent tracking
    length = [1] * n
    parent = [-1] * n
    best_end = 0
    for i in range(n):
        for j in range(i):
            if bases[j] < bases[i] and length[j] + 1 > length[i]:
                length[i] = length[j] + 1
                parent[i] = j
        if length[i] > length[best_end]:
            best_end = i

    # Reconstruct
    indices = []
    cur = best_end
    while cur != -1:
        indices.append(cur)
        cur = parent[cur]
    indices.reverse()
    return [matches[i] for i in indices]


def split_into_articles(text: str) -> list[dict]:
    """Split a normalized law into article-level chunks.

    Returns:
        List of dicts with keys ``article_number``, ``header``, ``text``.
        If no article headers are found, returns a single ``"full"`` chunk.
    """
    if not text or not text.strip():
        return []

    matches = list(_ARTICLE_HEADER_RE.finditer(text))
    matches = _filter_by_dominant_style(matches)

    if not matches:
        return [
            {
                "article_number": "full",
                "header": "",
                "text": text.strip(),
            }
        ]

    chunks: list[dict] = []

    # Preamble = everything before the first article header
    preamble = text[: matches[0].start()].strip()
    if len(preamble) >= _MIN_PREAMBLE_CHARS:
        chunks.append(
            {
                "article_number": "preambul",
                "header": "",
                "text": preamble,
            }
        )

    # Each article = from its header to the start of the next header
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if not body:
            continue  # empty article body, skip
        chunks.append(
            {
                "article_number": _normalize_number(m.group("num"), m.group("unic")),
                "header": m.group(0).strip(),
                "text": body,
            }
        )

    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 -m src.article_splitter <path-to-txt>", file=sys.stderr)
        sys.exit(1)

    # Use load_one so normalization is applied (matches real pipeline use)
    from .law_loader import load_one

    law_id, text, _meta = load_one(Path(sys.argv[1]))
    chunks = split_into_articles(text)

    print(f"Law: {law_id}")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        preview = c["text"][:80].replace("\n", " ")
        print(f"  [{c['article_number']:<10}] {len(c['text']):>5} chars | {preview}...")
