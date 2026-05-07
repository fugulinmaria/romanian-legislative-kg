"""Similarity metrics between a document subgraph and the ontology layer.

Inputs are triples DataFrames with columns: head, relation, tail
(law_id / article_number optional, ignored by the metrics).

Metrics:
    - relation_jaccard:        overlap of relation vocabularies actually used
    - relation_distribution_cosine: cosine on relation-frequency vectors
    - node_jaccard:            overlap of canonical entities
    - triple_jaccard:          overlap of full (head, relation, tail) triples
    - schema_signature_jaccard: overlap of (relation) usage patterns weighted
                               by anchor entities shared with the ontology
    - axiom_conformance:       1 - violations / total_triples (from ontology axioms)

A combined `compliance_score` aggregates the above with sensible weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from .config import (
    LEGISLATIVE_ASYMMETRIC_RELATIONS,
    LEGISLATIVE_CONSTRAINTS,
    LEGISLATIVE_FUNCTIONAL_RELATIONS,
    LEGISLATIVE_IRREFLEXIVE_RELATIONS,
)
from .relation_vocabulary import ALLOWED_RELATIONS

# ----------------------------- helpers ---------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _cosine(v1: dict, v2: dict) -> float:
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _triple_set(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty:
        return set()
    return set(zip(df["head"].astype(str), df["relation"].astype(str), df["tail"].astype(str)))


def _node_set(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(df["head"].astype(str)) | set(df["tail"].astype(str))


def _relation_set(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(df["relation"].astype(str))


def _relation_distribution(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    return df["relation"].value_counts().to_dict()


# ----------------------------- axiom checks ----------------------------------


def count_axiom_violations(df: pd.DataFrame) -> dict[str, int]:
    """Lightweight, side-effect-free re-implementation of the key axioms.

    Mirrors `LegislativeOntologyReasoner` but returns counts instead of
    printing. Used as a numeric compliance signal.
    """
    if df.empty:
        return {
            "functional": 0,
            "asymmetric": 0,
            "irreflexive": 0,
            "domain": 0,
            "vocabulary": 0,
            "total": 0,
        }

    # Functional: same (head, rel) -> multiple tails
    func = df[df["relation"].isin(LEGISLATIVE_FUNCTIONAL_RELATIONS)]
    func_violations = (
        func.groupby(["head", "relation"])["tail"].nunique().pipe(lambda s: int((s > 1).sum()))
        if not func.empty
        else 0
    )

    # Asymmetric: (h, r, t) and (t, r, h) both present
    asym = df[df["relation"].isin(LEGISLATIVE_ASYMMETRIC_RELATIONS)]
    asym_violations = 0
    if not asym.empty:
        edges = set(zip(asym["head"], asym["tail"], asym["relation"]))
        seen: set[tuple[str, str, str]] = set()
        for h, t, r in edges:
            if h == t:
                continue  # counted as irreflexive
            if (t, h, r) in edges:
                key = (min(h, t), max(h, t), r)
                if key not in seen:
                    seen.add(key)
                    asym_violations += 1

    # Irreflexive: head == tail
    irr = df[df["relation"].isin(LEGISLATIVE_IRREFLEXIVE_RELATIONS)]
    irr_violations = int((irr["head"] == irr["tail"]).sum()) if not irr.empty else 0

    # Domain: tail must be in allowed set
    domain_violations = 0
    for key, allowed in LEGISLATIVE_CONSTRAINTS.items():
        rel = key.replace("_domain", "")
        sub = df[df["relation"] == rel]
        if sub.empty:
            continue
        domain_violations += int((~sub["tail"].isin(allowed)).sum())

    # Vocabulary: relations outside the closed set (should be filtered upstream,
    # but we report it for safety).
    vocab_violations = int((~df["relation"].isin(ALLOWED_RELATIONS)).sum())

    total = (
        func_violations + asym_violations + irr_violations + domain_violations + vocab_violations
    )

    return {
        "functional": func_violations,
        "asymmetric": asym_violations,
        "irreflexive": irr_violations,
        "domain": domain_violations,
        "vocabulary": vocab_violations,
        "total": total,
    }


# ----------------------------- main API --------------------------------------


@dataclass
class SimilarityReport:
    doc_id: str
    n_triples: int
    n_dropped_relations: int
    relation_jaccard: float
    relation_distribution_cosine: float
    node_jaccard: float
    triple_jaccard: float
    axiom_conformance: float
    axiom_violations: dict[str, int]
    compliance_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_to_ontology(
    doc_df: pd.DataFrame,
    onto_df: pd.DataFrame,
    doc_id: str,
    n_dropped_relations: int = 0,
    weights: dict[str, float] | None = None,
) -> SimilarityReport:
    """Compute similarity metrics between a document subgraph and the
    ontology layer (the reference KB triples).

    The `compliance_score` is a weighted average in [0, 1]; higher == more
    aligned with the ontology and ontologically valid.
    """
    weights = weights or {
        "relation_jaccard": 0.20,
        "relation_distribution_cosine": 0.15,
        "node_jaccard": 0.10,
        "triple_jaccard": 0.05,
        "axiom_conformance": 0.50,
    }

    rel_j = _jaccard(_relation_set(doc_df), _relation_set(onto_df))
    rel_cos = _cosine(_relation_distribution(doc_df), _relation_distribution(onto_df))
    node_j = _jaccard(_node_set(doc_df), _node_set(onto_df))
    trip_j = _jaccard(_triple_set(doc_df), _triple_set(onto_df))

    violations = count_axiom_violations(doc_df)
    n = max(len(doc_df), 1)
    conformance = max(0.0, 1.0 - violations["total"] / n)

    metrics = {
        "relation_jaccard": rel_j,
        "relation_distribution_cosine": rel_cos,
        "node_jaccard": node_j,
        "triple_jaccard": trip_j,
        "axiom_conformance": conformance,
    }
    score = sum(weights[k] * metrics[k] for k in metrics)

    return SimilarityReport(
        doc_id=doc_id,
        n_triples=len(doc_df),
        n_dropped_relations=n_dropped_relations,
        relation_jaccard=rel_j,
        relation_distribution_cosine=rel_cos,
        node_jaccard=node_j,
        triple_jaccard=trip_j,
        axiom_conformance=conformance,
        axiom_violations=violations,
        compliance_score=score,
    )
