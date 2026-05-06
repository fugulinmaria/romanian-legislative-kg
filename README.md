# Romanian Legislative Knowledge Graph

A modular pipeline for extracting, storing, and reasoning over a knowledge
graph built from real Romanian legislation, using hybrid regex + LLM triple
extraction and a closed relation vocabulary.

## Features

- **Real corpus** — 8 normative acts (laws, government emergency ordinances,
  government decisions) downloaded from
  [legislatie.just.ro](https://legislatie.just.ro), stored in
  [data/raw_laws/](data/raw_laws/).
- **Hybrid extraction** — deterministic regex for legislative citations
  (bidirectional trigger search, with self-citation filtering) + the
  `gemma2:9b` LLM for richer semantic relations, with windowing
  (4000 chars / 200 overlap) for long articles.
- **Closed vocabulary** — 16 canonical relations (see below); any relation
  invented by the LLM is normalized via a synonym map or filtered out.
- **Anaphora resolution** — `prezenta lege`, `codul`, `hotărârea` are
  rewritten to the current `law_id` (in both `head` and `tail`).
- **Entity canonicalization** — NFC, diacritics, whitespace, casing — surface
  variants collapse to a single graph node.
- **Provenance** — every triple keeps `law_id` + `article_number` through the
  whole stack (KB, NetworkX graph, ontology reasoner).
- **Vector store** — ChromaDB persisted at `output/legislative_knowledge_db/`
  with `nomic-embed-text` embeddings.
- **Ontology reasoning** — 6 axioms (functional, asymmetric, irreflexive,
  symmetric, domain, transitive) with provenance annotation on violations.
- **Ollama health check** — the pipeline fails fast with a clear message if
  the server or models are missing.

---

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running (`ollama serve`)
- Models pulled:
  ```bash
  ollama pull gemma2:9b
  ollama pull nomic-embed-text
  ```

### Install

```bash
git clone <repo> && cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Standard run (appends to existing KB)
python main.py

# Clean run (wipes CSVs + Chroma DB first)
python main.py --fresh

# Quick smoke test
python main.py --fresh --max-laws 1 --max-articles 3

# Override models / windowing parameters
python main.py --llm-model llama3.1:8b --llm-window 3000 --llm-overlap 150

# Synthetic mode (no real corpus)
python main.py --synthetic
```

See `python main.py --help` for the full flag list.

### Validate results

```bash
python scripts/validate_pipeline.py
python scripts/validate_pipeline.py --laws lege_24_2000 oug_13_2026
```

---

## Project layout

```
Code/
├── data/raw_laws/              # Normative acts (.txt + .meta.json) + SOURCES.md
├── scripts/
│   ├── generate_meta.py        # Auto-generate .meta.json files
│   └── validate_pipeline.py    # Metric + ontology validation
├── src/
│   ├── config.py                  # Models, paths, ontology configuration
│   ├── text_normalizer.py         # NFC + diacritics + header stripping
│   ├── law_loader.py              # LawRecord = (law_id, text, meta)
│   ├── article_splitter.py        # Article segmentation (LIS anti-noise)
│   ├── relation_vocabulary.py     # 16 canonical relations + synonyms + prompt
│   ├── cross_reference_extractor.py  # Regex pre-pass (bidirectional)
│   ├── pronoun_resolver.py        # Resolves "prezenta lege" etc.
│   ├── entity_canonicalizer.py    # Entity surface-form normalization
│   ├── llm_handler.py             # Ollama + windowing + parser + health check
│   ├── knowledge_base.py          # KB with provenance + ChromaDB
│   ├── graph_builder.py           # MultiDiGraph (from_pandas_edgelist)
│   ├── ontology.py                # 6 axioms with provenance annotation
│   ├── eda.py                     # EDA + visualizations
│   └── legislative_generator.py   # Synthetic generator (fallback)
├── output/                     # CSVs + Chroma DB (gitignored)
├── main.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--fresh` | – | Wipe CSVs in `output/` + Chroma DB before running |
| `--synthetic` | – | Use the synthetic generator instead of the real corpus |
| `--max-laws N` | all | Cap the number of acts processed |
| `--max-articles N` | all | Cap articles per act |
| `--llm-model` | `gemma2:9b` | Ollama LLM model name |
| `--embedding-model` | `nomic-embed-text` | Ollama embedding model name |
| `--llm-window` | `4000` | Max characters per LLM window |
| `--llm-overlap` | `200` | Overlap between LLM windows |

---

## Relation vocabulary (16)

| Relation | Meaning |
|---|---|
| `emis_de` | issuing authority (functional) |
| `promulgat_de` | promulgating authority (functional) |
| `publicat_în` | publication venue (functional) |
| `modifică` | amends another act (asymmetric, irreflexive, transitive) |
| `completează` | supplements (asymmetric, irreflexive) |
| `abroga` | repeals (asymmetric, irreflexive) |
| `introduce` | introduces a new article |
| `republică` | republishes |
| `aprobă` | approves |
| `face_referire_la` | generic reference |
| `intră_în_vigoare` | entry-into-force date / condition |
| `are_sediul_în` | entity headquarters |
| `responsabil_pentru` | responsibility |
| `colaborează_cu` | collaboration (symmetric) |
| `se_aplică` | scope of application |
| `transpune` | transposes EU directive / regulation |

The associated ontology constants (functional / asymmetric / etc.) live in
[src/config.py](src/config.py).

---

## Corpus

8 normative acts — full details in
[data/raw_laws/SOURCES.md](data/raw_laws/SOURCES.md).

| `law_id` | Act | Year |
|---|---|---|
| `lege_53_2003` | Labour Code (republished) | 2003 / 2011 |
| `lege_190_2018` | GDPR application law | 2018 |
| `lege_24_2000` | Legislative drafting norms | 2000 / 2004 |
| `oug_156_2024` | Fiscal-budgetary measures 2025 | 2024 |
| `oug_13_2026` | Fiscal-budgetary amendments | 2026 |
| `oug_5_2026` | Credit-institutions amendments | 2026 |
| `hg_214_2026` | SPP inventory value updates | 2026 |
| `hg_24_2026` | Sub-prefect dismissal | 2026 |

---

## Outputs

In `output/`:

- `legislative_triples.csv` — triples `(head, relation, tail, law_id, article_number)`.
- `legislative_corpus.csv` — normalized law texts.
- `knowledge_base_export.csv` — full export.
- `legislative_knowledge_db/` — persistent ChromaDB collection
  (`romanian_legislative_knowledge`).
- `legislative_relation_distribution.png`, `legislative_knowledge_graph.png`.

---

## Programmatic use

```python
from src import (
    LegislativeKnowledgeBase, build_graph_from_triples,
    LegislativeOntologyReasoner, resolve_pronouns, canonicalize_entities,
)

kb = LegislativeKnowledgeBase()
kb.load()

# Provenance-aware queries
kb.query_by_law("lege_53_2003")
kb.query_by_article("oug_13_2026", "5")

# Graph with provenance on edges
G = build_graph_from_triples(kb.triples_df)

# Ontology validation
LegislativeOntologyReasoner(kb.triples_df).run_all_tests()
```
