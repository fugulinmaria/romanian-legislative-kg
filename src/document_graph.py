"""Load a synthetic document description (JSON) into a triples DataFrame.

The JSON schema mirrors the KB triples schema so the same downstream code
(graph builder, ontology reasoner, similarity metrics) can be reused.

Schema:
    {
        "doc_id":   "<string>",
        "title":    "<string>",
        "triples":  [ {"head": ..., "relation": ..., "tail": ..., "article_number": ...}, ... ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .entity_canonicalizer import canonicalize_entities
from .knowledge_base import TRIPLE_COLS
from .relation_vocabulary import normalize_relation


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    triples_df: pd.DataFrame
    dropped_relations: list[str]  # relations rejected by the closed vocabulary


def load_document(path: str | Path) -> DocumentRecord:
    """Load a synthetic document JSON file into a `DocumentRecord`.

    - Normalizes relations against the closed vocabulary (drops unknown ones).
    - Canonicalizes head/tail entities.
    - Uses the document's own `doc_id` as the `law_id` provenance column.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    doc_id = data["doc_id"]
    title = data.get("title", doc_id)
    raw = data.get("triples", [])

    rows = []
    dropped: list[str] = []
    for t in raw:
        rel = normalize_relation(t.get("relation", ""))
        if rel is None:
            dropped.append(t.get("relation", ""))
            continue
        rows.append(
            {
                "head": t["head"],
                "relation": rel,
                "tail": t["tail"],
                "law_id": doc_id,
                "article_number": str(t.get("article_number", "")),
            }
        )

    df = pd.DataFrame(rows, columns=TRIPLE_COLS)
    df = canonicalize_entities(df)
    return DocumentRecord(doc_id=doc_id, title=title, triples_df=df, dropped_relations=dropped)
