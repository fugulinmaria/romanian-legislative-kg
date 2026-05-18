"""Tests for source_method and confidence metadata on extracted triples."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import law_id_resolver
from src.config import CONFIDENCE_LLM, CONFIDENCE_REGEX
from src.cross_reference_extractor import extract_cross_references
from src.graph_builder import build_graph_from_triples
from src.knowledge_base import TRIPLE_COLS, LegislativeKnowledgeBase
from src.llm_handler import TripleExtractor


@pytest.fixture(autouse=True)
def _reset_registry():
    law_id_resolver._reset_for_tests()
    law_id_resolver.register_law(
        {
            "law_id": "lege_53_2003",
            "tip_act": "Lege",
            "numar": "53",
            "an": "2003",
            "aliases": ["Codul muncii"],
        }
    )
    yield
    law_id_resolver._reset_for_tests()


# ---------------------------------------------------------------------------
# cross_reference_extractor
# ---------------------------------------------------------------------------


def test_regex_triples_have_source_method_and_confidence():
    text = "Prezenta ordonanță modifică Legea nr. 53/2003 privind Codul muncii."
    df = extract_cross_references(text, current_law_id="oug_13_2026")
    assert not df.empty
    assert "source_method" in df.columns
    assert "confidence" in df.columns
    assert (df["source_method"] == "regex").all()
    assert (df["confidence"] == CONFIDENCE_REGEX).all()


def test_regex_empty_result_has_columns():
    df = extract_cross_references("Text fără referințe legislative.", current_law_id="oug_13_2026")
    assert df.empty
    assert "source_method" in df.columns
    assert "confidence" in df.columns


def test_compound_verb_expression_emits_two_triples():
    text = (
        "Prezenta ordonanță pentru modificarea și completarea "
        "Legea nr. 53/2003 intră în vigoare astăzi."
    )
    df = extract_cross_references(text, current_law_id="oug_13_2026")
    assert len(df) == 2
    relations = set(df["relation"].tolist())
    assert "modifică" in relations
    assert "completează" in relations
    assert (df["tail"] == "lege_53_2003").all()


# ---------------------------------------------------------------------------
# TripleExtractor (LLM) — mocked to avoid requiring Ollama
# ---------------------------------------------------------------------------


def _make_extractor_with_mock_response(raw_response: str) -> TripleExtractor:
    mock_llm = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = raw_response
    mock_llm.create_chain.return_value = chain
    return TripleExtractor(llm_handler=mock_llm)


def test_llm_triples_have_source_method_and_confidence():
    extractor = _make_extractor_with_mock_response("[lege_53_2003 | modifica | oug_13_2026]")
    df = extractor.extract_from_romanian_text("Articol de test.")
    assert not df.empty
    assert "source_method" in df.columns
    assert "confidence" in df.columns
    assert (df["source_method"] == "llm").all()
    assert (df["confidence"] == CONFIDENCE_LLM).all()


def test_llm_empty_result_has_no_metadata_columns():
    extractor = _make_extractor_with_mock_response("")
    df = extractor.extract_from_romanian_text("Text fără triplete.")
    assert df.empty
    assert "source_method" not in df.columns or df.empty


# ---------------------------------------------------------------------------
# Merge dedup: regex wins over LLM on the same (head, relation, tail)
# ---------------------------------------------------------------------------


def _make_triple_df(source_method: str, confidence: float) -> pd.DataFrame:
    return pd.DataFrame(
        [["lege_53_2003", "modifica", "oug_13_2026", source_method, confidence]],
        columns=["head", "relation", "tail", "source_method", "confidence"],
    )


def test_regex_wins_on_conflict_after_sort_dedup():
    regex_df = _make_triple_df("regex", CONFIDENCE_REGEX)
    llm_df = _make_triple_df("llm", CONFIDENCE_LLM)

    merged = pd.concat([regex_df, llm_df], ignore_index=True)
    merged = merged.sort_values("confidence", ascending=False)
    merged = merged.drop_duplicates(subset=["head", "relation", "tail"])

    assert len(merged) == 1
    assert merged.iloc[0]["source_method"] == "regex"
    assert merged.iloc[0]["confidence"] == CONFIDENCE_REGEX


def test_distinct_triples_both_kept():
    regex_df = _make_triple_df("regex", CONFIDENCE_REGEX)
    llm_df = pd.DataFrame(
        [["lege_53_2003", "emis_de", "Parlamentul României", "llm", CONFIDENCE_LLM]],
        columns=["head", "relation", "tail", "source_method", "confidence"],
    )

    merged = pd.concat([regex_df, llm_df], ignore_index=True)
    merged = merged.sort_values("confidence", ascending=False)
    merged = merged.drop_duplicates(subset=["head", "relation", "tail"])

    assert len(merged) == 2


# ---------------------------------------------------------------------------
# knowledge_base
# ---------------------------------------------------------------------------


def test_triple_cols_includes_metadata():
    assert "source_method" in TRIPLE_COLS
    assert "confidence" in TRIPLE_COLS


def test_knowledge_base_stores_and_retrieves_metadata():
    kb = LegislativeKnowledgeBase.__new__(LegislativeKnowledgeBase)
    kb.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
    kb.corpus = []

    incoming = pd.DataFrame(
        [
            [
                "lege_53_2003",
                "modifica",
                "oug_13_2026",
                "lege_53_2003",
                "1",
                "regex",
                CONFIDENCE_REGEX,
            ],
            ["lege_53_2003", "emis_de", "Parlament", "lege_53_2003", "2", "llm", CONFIDENCE_LLM],
        ],
        columns=TRIPLE_COLS,
    )
    kb.add_triples(incoming)

    assert "source_method" in kb.triples_df.columns
    assert "confidence" in kb.triples_df.columns

    regex_rows = kb.triples_df[kb.triples_df["source_method"] == "regex"]
    llm_rows = kb.triples_df[kb.triples_df["source_method"] == "llm"]
    assert len(regex_rows) == 1
    assert len(llm_rows) == 1
    assert float(regex_rows.iloc[0]["confidence"]) == CONFIDENCE_REGEX
    assert float(llm_rows.iloc[0]["confidence"]) == CONFIDENCE_LLM


def test_knowledge_base_add_triples_fills_missing_metadata():
    kb = LegislativeKnowledgeBase.__new__(LegislativeKnowledgeBase)
    kb.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
    kb.corpus = []

    incoming = pd.DataFrame(
        [["lege_53_2003", "modifica", "oug_13_2026", "lege_53_2003", "1"]],
        columns=["head", "relation", "tail", "law_id", "article_number"],
    )
    kb.add_triples(incoming)

    assert "source_method" in kb.triples_df.columns
    assert "confidence" in kb.triples_df.columns


def test_get_triple_sources_includes_metadata():
    kb = LegislativeKnowledgeBase.__new__(LegislativeKnowledgeBase)
    kb.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
    kb.corpus = []

    incoming = pd.DataFrame(
        [
            [
                "lege_53_2003",
                "modifica",
                "oug_13_2026",
                "lege_53_2003",
                "1",
                "regex",
                CONFIDENCE_REGEX,
            ]
        ],
        columns=TRIPLE_COLS,
    )
    kb.add_triples(incoming)

    sources = kb.get_triple_sources("lege_53_2003", "modifica", "oug_13_2026")
    assert len(sources) == 1
    assert sources[0]["source_method"] == "regex"
    assert float(sources[0]["confidence"]) == CONFIDENCE_REGEX


# ---------------------------------------------------------------------------
# graph_builder
# ---------------------------------------------------------------------------


def test_graph_edges_have_source_method_and_confidence():
    df = pd.DataFrame(
        [
            ["lege_53_2003", "modifica", "oug_13_2026", "regex", CONFIDENCE_REGEX],
            ["lege_53_2003", "emis_de", "Parlament", "llm", CONFIDENCE_LLM],
        ],
        columns=["head", "relation", "tail", "source_method", "confidence"],
    )
    g = build_graph_from_triples(df, verbose=False)

    edges = list(g.edges(data=True))
    for _, _, attrs in edges:
        assert "source_method" in attrs
        assert "confidence" in attrs
