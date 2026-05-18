"""Tests for law_id resolution and graph collapsing."""

import pandas as pd
import pytest

from src import law_id_resolver
from src.cross_reference_extractor import extract_cross_references
from src.entity_canonicalizer import canonicalize, canonicalize_entities
from src.graph_builder import build_graph_from_triples


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


def test_resolve_variants_to_same_law_id():
    forms = [
        "Legea nr. 53/2003",
        "LEGEA 53/2003",
        "Legea nr.53/2003",
        "  legea nr. 53 / 2003  ",
        "Codul muncii",
        "codul MUNCII",
        "lege_53_2003",
    ]
    for f in forms:
        assert law_id_resolver.resolve(f) == "lege_53_2003", f


def test_synthesize_unknown_well_formed_citation():
    assert law_id_resolver.resolve("Legea nr. 999/1999") == "lege_999_1999"
    assert law_id_resolver.resolve("O.U.G. nr. 89/2025") == "oug_89_2025"
    assert law_id_resolver.resolve("Hotararea Guvernului nr. 1.705/2006") == "hg_1705_2006"


def test_resolve_returns_none_for_bogus():
    assert law_id_resolver.resolve("random phrase") is None
    assert law_id_resolver.resolve("") is None


def test_code_names_resolve():
    assert law_id_resolver.resolve("Codul fiscal") == "lege_227_2015"
    assert law_id_resolver.resolve("Constituția României") == "constitutia_romaniei"


def test_canonicalize_collapses_to_law_id():
    assert canonicalize("Legea nr. 53/2003") == "lege_53_2003"
    assert canonicalize("Codul muncii") == "lege_53_2003"
    assert canonicalize("lege_53_2003") == "lege_53_2003"


def test_canonicalize_entities_dataframe():
    df = pd.DataFrame(
        [
            ["oug_13_2026", "modifica", "Legea nr. 53/2003"],
            ["oug_13_2026", "modifica", "Codul muncii"],
            ["oug_13_2026", "modifica", "lege_53_2003"],
        ],
        columns=["head", "relation", "tail"],
    )
    out = canonicalize_entities(df)
    assert out["tail"].nunique() == 1
    assert out["tail"].iloc[0] == "lege_53_2003"


def test_cross_reference_extractor_emits_law_ids():
    text = (
        "Articolul 1. Prezenta ordonanță modifică Legea nr. 53/2003 privind Codul muncii "
        "și abrogă Hotărârea Guvernului nr. 1.705/2006."
    )
    df = extract_cross_references(text, current_law_id="oug_13_2026")
    tails = set(df["tail"])
    assert "lege_53_2003" in tails
    assert "hg_1705_2006" in tails
    # Surface forms should be gone
    assert not any("Legea nr." in t for t in tails)


def test_graph_collapses_surface_variants():
    df = pd.DataFrame(
        [
            ["oug_13_2026", "modifica", "Legea nr. 53/2003"],
            ["oug_13_2026", "modifica", "Codul muncii"],
            ["oug_13_2026", "modifica", "lege_53_2003"],
        ],
        columns=["head", "relation", "tail"],
    )
    df = canonicalize_entities(df)
    g = build_graph_from_triples(df, verbose=False)
    assert "lege_53_2003" in g.nodes
    assert g.nodes["lege_53_2003"]["node_type"] == "law"
    assert g.nodes["lege_53_2003"]["tip_act"] == "Lege"
    # Only one tail node, not three
    assert g.number_of_nodes() == 2
