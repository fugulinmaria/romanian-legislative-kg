"""
Romanian Legislative Knowledge Graph Analysis Pipeline
Main script for generating, extracting, storing, and reasoning over Romanian legislative knowledge.
"""

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
)
from src.config import LLM_TEMPERATURE_EXTRACTION


def main():
    """Main execution pipeline for Romanian legislative knowledge graph analysis."""

    print("\n" + "=" * 80)
    print(" ROMANIAN LEGISLATIVE KNOWLEDGE GRAPH ANALYSIS PIPELINE")
    print("=" * 80)

    # ============================================================================
    # STEP 1: Initialize Components
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 1: INITIALIZATION")
    print("-" * 80)

    llm_handler = LLMHandler(temperature=LLM_TEMPERATURE_EXTRACTION, verbose=True)
    extractor = TripleExtractor(llm_handler=llm_handler)
    generator = RomanianLegislativeGenerator()
    kb = LegislativeKnowledgeBase()

    embeddings_handler = EmbeddingsHandler(verbose=True)
    vector_store_handler = VectorStoreHandler(embeddings_handler=embeddings_handler, verbose=True)
    vector_store = vector_store_handler.get_store()

    # ============================================================================
    # STEP 2: Generate Legislative Texts
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 2: LEGISLATIVE TEXT GENERATION")
    print("-" * 80)

    print("\nGenerating diverse Romanian legislative texts...")
    laws = generator.generate_batch(
        count=5, law_types=["simple", "complex", "amendment", "emergency"]
    )

    print(f"\n[OK] Generated {len(laws)} legislative documents")
    for law_id, _ in laws:
        print(f"  - {law_id}")

    # ============================================================================
    # STEP 3: Extract Triples from Legislative Texts
    # ============================================================================
    print("\n" + "-" * 80)
    print(" STEP 3: KNOWLEDGE EXTRACTION")
    print("-" * 80)

    all_triples = []

    for law_id, law_text in laws:
        print(f"\nProcessing: {law_id}")
        print(f"Text preview: {law_text[:150]}...")

        triples_df = extractor.extract_from_romanian_text(law_text)

        if not triples_df.empty:
            print(f"  [OK] Extracted {len(triples_df)} triples")
            all_triples.append(triples_df)
            kb.add_law_to_corpus(law_id, law_text)
            kb.add_triples(triples_df)
        else:
            print("  [WARN] No triples extracted")

        vector_store.add_texts(
            texts=[law_text], metadatas=[{"law_id": law_id, "source": "generated"}], ids=[law_id]
        )
        print("  [OK] Added to vector store")

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
        "modificare și completare legislație",
        "digitalizare și tehnologie",
        "mediu și protecție",
    ]

    for query in search_queries:
        print(f"\n[SEARCH] Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)

        if results:
            print(f"  Found {len(results)} similar laws:")
            for i, doc in enumerate(results, 1):
                law_id = doc.metadata.get("law_id", "unknown")
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"    {i}. {law_id}: {preview}...")
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
