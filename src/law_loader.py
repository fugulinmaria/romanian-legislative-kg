"""
Loader for the real Romanian legislative corpus.

Walks a directory of paired ``<law_id>.txt`` + ``<law_id>.meta.json`` files,
applies the Romanian text normalizer, and returns records ready to be fed into
the existing pipeline (corpus, vector store, triple extractor).

A ``LawRecord`` is ``(law_id, normalized_text, metadata)`` — the first two
elements match the shape of ``RomanianLegislativeGenerator.generate_batch()``,
so swapping the source in ``main.py`` is a one-line change.

Usage (CLI):
    python3 -m src.law_loader
    python3 -m src.law_loader data/raw_laws
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

from .law_id_resolver import register_law
from .text_normalizer import normalize_ro

LawRecord = Tuple[str, str, dict]

DEFAULT_DIR = "data/raw_laws"


def _read_meta(meta_path: Path, law_id: str) -> dict:
    """Read .meta.json sidecar; return a minimal fallback dict if missing."""
    if not meta_path.exists():
        print(f"[WARN] No metadata file for '{law_id}', using fallback")
        return {"law_id": law_id}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[WARN] Invalid JSON in {meta_path.name}: {e}; using fallback")
        return {"law_id": law_id}
    meta.setdefault("law_id", law_id)
    return meta


def load_one(txt_path: str | Path) -> LawRecord:
    """Load and normalize a single law file plus its metadata sidecar."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    law_id = txt_path.stem
    raw = txt_path.read_text(encoding="utf-8")
    text = normalize_ro(raw)
    meta = _read_meta(txt_path.with_suffix(".meta.json"), law_id)
    register_law(meta)
    return law_id, text, meta


def load_real_laws(directory: str | Path = DEFAULT_DIR) -> list[LawRecord]:
    """Load all real laws from a directory.

    Args:
        directory: Folder containing ``*.txt`` + ``*.meta.json`` pairs.

    Returns:
        List of ``(law_id, normalized_text, metadata)`` tuples, sorted by
        ``law_id`` for deterministic ordering. Empty ``.txt`` files are skipped
        with a warning.
    """
    base = Path(directory).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    records: list[LawRecord] = []
    txt_files = sorted(base.glob("*.txt"))

    for txt_path in txt_files:
        if txt_path.stat().st_size == 0:
            print(f"[SKIP] Empty file: {txt_path.name}")
            continue
        try:
            records.append(load_one(txt_path))
        except Exception as e:  # noqa: BLE001 - keep loader robust during dev
            print(f"[ERROR] Failed to load {txt_path.name}: {e}")

    print(f"Loaded {len(records)} real law(s) from {base}")
    return records


def _print_summary(records: list[LawRecord]) -> None:
    """Pretty-print a one-row-per-law summary table."""
    print()
    print(f"{'law_id':<22} {'tip_act':<22} {'nr/an':<12} {'chars':>7}  preview")
    print("-" * 100)
    for law_id, text, meta in records:
        nr_an = f"{meta.get('numar', '?')}/{meta.get('an', '?')}"
        preview = text[:60].replace("\n", " ")
        print(
            f"{law_id:<22} {meta.get('tip_act', ''):<22} {nr_an:<12} {len(text):>7}  {preview}..."
        )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    recs = load_real_laws(target)
    _print_summary(recs)
