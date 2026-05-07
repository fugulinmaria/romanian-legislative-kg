"""Compare synthetic document subgraphs against the ontology / KB layer.

Usage:
    python scripts/compare_documents.py
    python scripts/compare_documents.py --docs data/synthetic_docs/doc_valid.json \
                                        data/synthetic_docs/doc_invalid.json
    python scripts/compare_documents.py --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    LegislativeKnowledgeBase,
    LegislativeOntologyReasoner,
    build_graph_from_triples,
    canonicalize_entities,
)
from src.config import OUTPUT_DIR  # noqa: E402
from src.document_graph import load_document  # noqa: E402
from src.similarity import compare_to_ontology  # noqa: E402

DEFAULT_DOCS = [
    ROOT / "data" / "synthetic_docs" / "doc_valid.json",
    ROOT / "data" / "synthetic_docs" / "doc_invalid.json",
]


def _print_report(report) -> None:
    d = report.to_dict()
    print(f"\n--- {d['doc_id']} ---")
    print(
        f"  triples: {d['n_triples']}|dropped (out-of-vocab) relations: {d['n_dropped_relations']}"  # noqa: E501
    )
    print(f"  relation_jaccard            : {d['relation_jaccard']:.3f}")
    print(f"  relation_distribution_cosine: {d['relation_distribution_cosine']:.3f}")
    print(f"  node_jaccard                : {d['node_jaccard']:.3f}")
    print(f"  triple_jaccard              : {d['triple_jaccard']:.3f}")
    print(f"  axiom_conformance           : {d['axiom_conformance']:.3f}")
    print(f"  axiom_violations            : {d['axiom_violations']}")
    print(f"  COMPLIANCE SCORE            : {d['compliance_score']:.3f}")


def _maybe_plot(graphs: dict, reports: list, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("matplotlib / networkx not available, skipping plot.")
        return

    n = len(graphs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, G) in zip(axes, graphs.items()):
        pos = nx.spring_layout(G, seed=42, k=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=350, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, arrows=True)
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6)
        ax.set_title(f"{name}\nnodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved graph figure to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs",
        nargs="+",
        default=[str(p) for p in DEFAULT_DOCS],
        help="JSON document files to compare against the KB.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render the ontology + document subgraphs side-by-side.",
    )
    parser.add_argument(
        "--no-axiom-print",
        action="store_true",
        help="Skip the verbose ontology reasoner output for each document.",
    )
    args = parser.parse_args()

    # ---- 1. Ontology layer (existing KB) ----
    kb = LegislativeKnowledgeBase()
    if kb.triples_df.empty:
        print("ERROR: KB is empty. Run `python main.py` first to build the corpus.")
        return 1
    onto_df = canonicalize_entities(kb.triples_df)
    print(
        f"\nOntology layer: {len(onto_df)} triples, "
        f"{onto_df['head'].nunique() + onto_df['tail'].nunique()} raw entity refs."
    )

    G_onto = build_graph_from_triples(onto_df, verbose=False)
    graphs = {"ontology (KB)": G_onto}

    # ---- 2. Per-document compare ----
    reports = []
    for doc_path in args.docs:
        doc = load_document(doc_path)
        print(
            f"\nLoaded {doc.doc_id}: {len(doc.triples_df)} triples "
            f"({len(doc.dropped_relations)} dropped: {doc.dropped_relations})"
        )

        if not args.no_axiom_print:
            LegislativeOntologyReasoner(doc.triples_df).run_all_tests()

        rep = compare_to_ontology(
            doc_df=doc.triples_df,
            onto_df=onto_df,
            doc_id=doc.doc_id,
            n_dropped_relations=len(doc.dropped_relations),
        )
        _print_report(rep)
        reports.append(rep)

        graphs[doc.doc_id] = build_graph_from_triples(doc.triples_df, verbose=False)

    # ---- 3. Save report ----
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "document_similarity_report.json"
    report_path.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved report to {report_path}")

    # CSV summary for the README / disertatie
    summary = pd.DataFrame(
        [
            {
                "doc_id": r.doc_id,
                "n_triples": r.n_triples,
                "relation_jaccard": round(r.relation_jaccard, 3),
                "rel_distrib_cosine": round(r.relation_distribution_cosine, 3),
                "node_jaccard": round(r.node_jaccard, 3),
                "triple_jaccard": round(r.triple_jaccard, 3),
                "axiom_conformance": round(r.axiom_conformance, 3),
                "violations_total": r.axiom_violations["total"],
                "compliance_score": round(r.compliance_score, 3),
            }
            for r in reports
        ]
    )
    summary_path = out_dir / "document_similarity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved CSV summary to {summary_path}")
    print("\n" + summary.to_string(index=False))

    # ---- 4. Optional plot ----
    if args.plot:
        _maybe_plot(graphs, reports, out_dir / "document_subgraphs.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
