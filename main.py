"""
Romanian Legislative Knowledge Graph Analysis Pipeline
Main script for generating, extracting, storing, and reasoning over Romanian legislative knowledge.
"""

import argparse
import os
import shutil

import pandas as pd

from src import (
    EmbeddingsHandler,
    KnowledgeGraphEDA,
    LegislativeKnowledgeBase,
    LegislativeOntologyReasoner,
    LLMHandler,
    RomanianLegislativeGenerator,
    TripleExtractor,
    VectorStoreHandler,
    build_graph_from_triples,
    canonicalize_entities,
    extract_cross_references,
    load_real_laws,
    resolve_pronouns,
    split_into_articles,
)
from src.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE_EXTRACTION,
    OUTPUT_DIR,
)

# ----------------------------------------------------------------------------
# Pipeline flags
# ----------------------------------------------------------------------------
USE_REAL_LAWS = True  # False -> use synthetic generator (legacy mode)
MAX_LAWS = None  # int or None; cap for fast smoke tests (e.g. 2)
MAX_ARTICLES_PER_LAW = None  # int or None; cap chunks per law during dev


_FRESH_TARGETS = [
    os.path.join(OUTPUT_DIR, "legislative_triples.csv"),
    os.path.join(OUTPUT_DIR, "legislative_corpus.csv"),
    os.path.join(OUTPUT_DIR, "knowledge_base_export.csv"),
]


def _wipe_outputs() -> None:
    """Remove persisted CSVs and the Chroma DB so the run starts clean."""
    print("[--fresh] Wiping previous outputs...")
    for path in _FRESH_TARGETS:
        if os.path.isfile(path):
            os.remove(path)
            print(f"  removed {path}")
    if os.path.isdir(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"  removed {CHROMA_DB_PATH}/")


def _load_input_laws(generator):
    """Return a list of (law_id, full_text, metadata) records for the pipeline."""
    if USE_REAL_LAWS:
        records = load_real_laws()
        if MAX_LAWS:
            records = records[:MAX_LAWS]
        return records

    # Legacy: synthetic generator returns (law_id, text); add empty meta
    print("\nGenerating diverse Romanian legislative texts...")
    laws = generator.generate_batch(
        count=5, law_types=["simple", "complex", "amendment", "emergency"]
    )
    return [(law_id, text, {"law_id": law_id}) for law_id, text in laws]


def main():
    """Main execution pipeline for Romanian legislative knowledge graph analysis."""
    global USE_REAL_LAWS, MAX_LAWS, MAX_ARTICLES_PER_LAW

    parser = argparse.ArgumentParser(description="Romanian Legislative Knowledge Graph pipeline.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe output/ CSVs and the Chroma DB before running.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the synthetic legislative generator instead of real laws.",
    )
    parser.add_argument(
        "--max-laws",
        type=int,
        default=MAX_LAWS,
        help="Cap the number of input laws (smoke-test mode).",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=MAX_ARTICLES_PER_LAW,
        help="Cap the number of articles per law.",
    )
    parser.add_argument(
        "--llm-model",
        default=LLM_MODEL,
        help=f"Ollama LLM model name (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Ollama embedding model name (default: {EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--llm-window",
        type=int,
        default=TripleExtractor.MAX_LLM_CHARS,
        help="Character cap per LLM extraction window.",
    )
    parser.add_argument(
        "--llm-overlap",
        type=int,
        default=TripleExtractor.LLM_OVERLAP,
        help="Overlap (chars) between LLM extraction windows.",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    USE_REAL_LAWS = not args.synthetic
    MAX_LAWS = args.max_laws
    MAX_ARTICLES_PER_LAW = args.max_articles
    TripleExtractor.MAX_LLM_CHARS = args.llm_window
    TripleExtractor.LLM_OVERLAP = args.llm_overlap

    if args.fresh:
        _wipe_outputs()

    print("\n" + "=" * 80)
    print(" ROMANIAN LEGISLATIVE KNOWLEDGE GRAPH ANALYSIS PIPELINE")
    print("=" * 80)
    print(f" Mode: {'REAL LAWS' if USE_REAL_LAWS else 'SYNTHETIC LAWS'}")

    # ============================================================================
    # STEP 1: Initialize Components
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 1: INITIALIZATION")
    print("-" * 80)

    llm_handler = LLMHandler(
        model=args.llm_model,
        temperature=LLM_TEMPERATURE_EXTRACTION,
        verbose=True,
    )
    llm_handler.health_check()
    extractor = TripleExtractor(llm_handler=llm_handler)
    generator = None if USE_REAL_LAWS else RomanianLegislativeGenerator()
    kb = LegislativeKnowledgeBase()

    embeddings_handler = EmbeddingsHandler(model=args.embedding_model, verbose=True)
    embeddings_handler.health_check()
    vector_store_handler = VectorStoreHandler(embeddings_handler=embeddings_handler, verbose=True)
    vector_store = vector_store_handler.get_store()

    # ============================================================================
    # STEP 2: Load Legislative Texts
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 2: LEGISLATIVE TEXT LOADING")
    print("-" * 80)

    laws = _load_input_laws(generator)

    print(f"\n[OK] Loaded {len(laws)} legislative documents")
    for law_id, _, _ in laws:
        print(f"  - {law_id}")

    # ============================================================================
    # STEP 3: Article-level Extraction & Indexing
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 3: KNOWLEDGE EXTRACTION (per article)")
    print("-" * 80)

    source_tag = "real" if USE_REAL_LAWS else "generated"
    all_triples = []

    for law_id, full_text, meta in laws:
        print(f"\nProcessing: {law_id}")
        kb.add_law_to_corpus(law_id, full_text)

        articles = split_into_articles(full_text)
        if MAX_ARTICLES_PER_LAW:
            articles = articles[:MAX_ARTICLES_PER_LAW]
        print(f"  Split into {len(articles)} chunks")

        law_triple_count = 0
        for art_idx, art in enumerate(articles, 1):
            art_num = art["article_number"]
            art_text = art["text"]
            print(
                f"  [{art_idx}/{len(articles)}] article {art_num} ({len(art_text)} chars)",
                end="",
                flush=True,
            )
            chunk_id = f"{law_id}::art_{art_num}"

            # 1. Deterministic regex pre-pass for cross-references
            regex_triples = extract_cross_references(art_text, current_law_id=law_id)

            # 2. LLM extraction for everything else
            llm_triples = extractor.extract_from_romanian_text(art_text)

            # 3. Merge + dedupe (regex wins on conflicts because it's deterministic)
            merged = pd.concat([regex_triples, llm_triples], ignore_index=True)
            if not merged.empty:
                # Rewrite anaphora ("prezenta lege", "hotărârea", "codul ...")
                # on both head and tail to the canonical law_id.
                merged = resolve_pronouns(merged, law_id)
                # Canonicalize surface forms (case, whitespace, diacritics)
                # so variants collapse to one graph node.
                merged = canonicalize_entities(merged)

                merged = merged.drop_duplicates(subset=["head", "relation", "tail"])
                merged = merged.assign(law_id=law_id, article_number=art_num)
                all_triples.append(merged)
                kb.add_triples(merged)
                law_triple_count += len(merged)

            # Index the article in the vector store with rich metadata
            chunk_meta = {
                "law_id": law_id,
                "article_number": str(art_num),
                "tip_act": meta.get("tip_act", ""),
                "numar": str(meta.get("numar", "")),
                "an": str(meta.get("an", "")),
                "titlu": meta.get("titlu", ""),
                "emitent": meta.get("emitent", ""),
                "data_publicare": meta.get("data_publicare", ""),
                "source_url": meta.get("source_url", ""),
                "source": source_tag,
            }
            try:
                vector_store.add_texts(texts=[art_text], metadatas=[chunk_meta], ids=[chunk_id])
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if "already exists" in err or "duplicate" in err.lower():
                    # Re-run: chunk already indexed, safe to skip
                    pass
                else:
                    print(f"  [WARN] vector store add failed for {chunk_id}: {err}")

            print(f" → {len(merged) if not merged.empty else 0} triples")

        print(f"  [OK] {law_triple_count} triples extracted, {len(articles)} chunks indexed")

    if all_triples:
        combined_triples = pd.concat(all_triples, ignore_index=True)
        print(f"\n[OK] Total triples extracted: {len(combined_triples)}")
    else:
        print("\n[WARN] No triples were extracted from any document")
        return

    # ============================================================================
    # STEP 4: Build Knowledge Graph
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 4: KNOWLEDGE GRAPH CONSTRUCTION")
    print("-" * 80)

    G = build_graph_from_triples(combined_triples, verbose=True)

    # ============================================================================
    # STEP 5: Exploratory Data Analysis
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 5: EXPLORATORY DATA ANALYSIS")
    print("-" * 80)

    eda = KnowledgeGraphEDA(combined_triples, G)
    eda.print_basic_stats()
    eda.print_graph_metrics()
    eda.print_relational_distribution(top_n=10)

    if len(combined_triples) > 0:
        print("\nGenerating visualizations...")
        eda.plot_relational_distribution(
            top_n=min(10, combined_triples["relation"].nunique()),
            filename="legislative_relation_distribution",
        )
        eda.plot_knowledge_graph_sample(max_nodes=50, filename="legislative_knowledge_graph")

    # ============================================================================
    # STEP 6: Ontological Reasoning
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 6: ONTOLOGICAL REASONING")
    print("-" * 80)

    reasoner = LegislativeOntologyReasoner(combined_triples)
    reasoner.run_all_tests()

    # ============================================================================
    # STEP 7: Query Knowledge Base
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 7: KNOWLEDGE BASE QUERIES")
    print("-" * 80)

    stats = kb.get_statistics()
    print("\nKnowledge Base Statistics:")
    print(f"  Total triples: {stats['total_triples']:,}")
    print(f"  Unique entities: {stats['unique_entities']:,}")
    print(f"  Unique relations: {stats['unique_relations']}")
    print(f"  Laws in corpus: {stats['total_laws_in_corpus']}")

    print("\nRelation Distribution:")
    for relation, count in list(stats["relation_distribution"].items())[:5]:
        print(f"  {relation}: {count}")

    print("\n--- Example Queries ---")

    modificari = kb.query_by_relation("modifică")
    if not modificari.empty:
        print(f"\nLaws that modify other laws: {len(modificari)}")
        for _, row in modificari.head(3).iterrows():
            print(f"  '{row['head']}' modifică '{row['tail']}'")

    # ============================================================================
    # STEP 7.5: Semantic Vector Search
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 7.5: SEMANTIC VECTOR SEARCH")
    print("-" * 80)

    print("\nPerforming semantic searches...")

    search_queries = [
        "protecția datelor cu caracter personal",
        "modificarea Codului fiscal",
        "raporturi de muncă și concediu",
    ]

    for query in search_queries:
        print(f"\n[SEARCH] Query: '{query}'")
        results = vector_store.similarity_search(query, k=3)

        if results:
            print(f"  Found {len(results)} similar chunks:")
            for i, doc in enumerate(results, 1):
                law_id = doc.metadata.get("law_id", "unknown")
                art = doc.metadata.get("article_number", "?")
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"    {i}. {law_id} (art. {art}): {preview}...")
        else:
            print("  No results found")

    # ============================================================================
    # STEP 8: Save Knowledge Base
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 8: PERSISTENCE")
    print("-" * 80)

    kb.save()
    kb.export_to_csv()

    # ============================================================================
    # PIPELINE COMPLETE
    # ============================================================================
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nOutputs saved to: output/")
    print("  - legislative_triples.csv (knowledge base)")
    print("  - legislative_corpus.csv (law texts)")
    print("  - knowledge_base_export.csv (export)")
    print("  - legislative_relation_distribution.png")
    print("  - legislative_knowledge_graph.png")
    print("  - legislative_knowledge_db/ (vector database)")


if __name__ == "__main__":
    main()
