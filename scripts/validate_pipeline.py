"""
Phase G — Pipeline validation script.

Loads the outputs produced by main.py and runs three tiers of checks:

  1. METRICS  — triple count, unique relations, distribution, relation coverage
  2. ONTOLOGY — functional / asymmetric / irreflexive violations (reuses
                 LegislativeOntologyReasoner from src/ontology.py)
  3. SPOT-CHECK — per-law triple sample + cross-reference audit

Usage:
    python3 scripts/validate_pipeline.py [--triples PATH] [--laws law_id ...]

Defaults:
    --triples output/legislative_triples.csv
    --laws    lege_53_2003  lege_190_2018  oug_156_2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ontology import LegislativeOntologyReasoner  # noqa: E402
from src.relation_vocabulary import ALLOWED_RELATIONS  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G pipeline validation")
    p.add_argument(
        "--triples",
        default=str(ROOT / "output" / "legislative_triples.csv"),
        help="Path to legislative_triples.csv",
    )
    p.add_argument(
        "--laws",
        nargs="+",
        default=["lege_53_2003", "lege_190_2018", "oug_156_2024"],
        help="law_ids to include in spot-check",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tier 1 — Metrics
# ---------------------------------------------------------------------------


def tier1_metrics(df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TIER 1 — METRICS")
    print(sep)

    print(f"\nTotal triples       : {len(df):,}")
    print(f"Unique head entities: {df['head'].nunique():,}")
    print(f"Unique tail entities: {df['tail'].nunique():,}")
    print(f"Unique relations    : {df['relation'].nunique()}")

    canonical = set(ALLOWED_RELATIONS)
    found = set(df["relation"].unique())
    covered = found & canonical
    rogue = found - canonical

    print(f"\nCanonical vocabulary : {len(canonical)} relations")
    print(f"Relations present    : {len(found)}")
    print(f"Covered canonical    : {len(covered)}")
    if rogue:
        print(f"OUT-OF-VOCAB (should be 0): {sorted(rogue)}")
    else:
        print("Out-of-vocab relations  : NONE ✅")

    print("\nRelation distribution:")
    dist = df["relation"].value_counts()
    for rel, cnt in dist.items():
        bar = "█" * min(40, cnt // max(1, len(df) // 400))
        print(f"  {rel:<25} {cnt:>5}  {bar}")

    # Per-law summary
    if "law_id" in df.columns:
        print("\nPer-law triple count:")
        for law_id, cnt in df.groupby("law_id").size().sort_values(ascending=False).items():
            print(f"  {law_id:<25} {cnt:>5}")


# ---------------------------------------------------------------------------
# Tier 2 — Ontology violations
# ---------------------------------------------------------------------------


def tier2_ontology(df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TIER 2 — ONTOLOGY VIOLATIONS")
    print(sep)

    # LegislativeOntologyReasoner expects only head/relation/tail
    triples_only = df[["head", "relation", "tail"]].copy()
    reasoner = LegislativeOntologyReasoner(triples_only)
    reasoner.run_all_tests()


# ---------------------------------------------------------------------------
# Tier 3 — Spot-check
# ---------------------------------------------------------------------------


def tier3_spotcheck(df: pd.DataFrame, law_ids: list[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TIER 3 — SPOT-CHECK")
    print(sep)

    id_col = "law_id" if "law_id" in df.columns else None

    for law_id in law_ids:
        print(f"\n--- {law_id} ---")

        if id_col:
            subset = df[df[id_col] == law_id]
        else:
            # Fallback: head starts with law_id
            subset = df[df["head"].str.startswith(law_id, na=False)]

        if subset.empty:
            print("  (no triples found for this law)")
            continue

        print(f"  Triples: {len(subset)}")
        print(f"  Unique relations: {subset['relation'].nunique()}")

        # Cross-references: any relation that points to another act
        xref_rels = {
            "modifică",
            "completează",
            "abroga",
            "face_referire_la",
            "aprobă",
            "republică",
            "transpune",
            "introduce",
        }
        xrefs = subset[subset["relation"].isin(xref_rels)]
        print(f"  Cross-references : {len(xrefs)}")

        # Sample 5 triples
        sample = subset.sample(min(5, len(subset)), random_state=42)
        print("\n  Sample triples:")
        for _, row in sample.iterrows():
            print(f"    [{row['head'][:40]:<40} | {row['relation']:<20} | {row['tail'][:50]}]")

        # Show all cross-references if any
        if not xrefs.empty:
            print("\n  All cross-references:")
            for _, row in xrefs.head(10).iterrows():
                print(f"    [{row['head'][:40]:<40} | {row['relation']:<20} | {row['tail'][:50]}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    triples_path = Path(args.triples)
    if not triples_path.exists():
        print(f"[ERROR] Triples file not found: {triples_path}")
        print("Run main.py first to generate output/legislative_triples.csv")
        sys.exit(1)

    df = pd.read_csv(triples_path)
    print(f"Loaded {len(df)} triples from {triples_path}")

    tier1_metrics(df)
    tier2_ontology(df)
    tier3_spotcheck(df, args.laws)

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
