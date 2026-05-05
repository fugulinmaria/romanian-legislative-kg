# Graf de Cunoaștere Legislativă Românească

Un framework modular pentru extragerea, stocarea și raționamentul asupra
grafurilor de cunoaștere construite din acte normative românești reale,
folosind extragere de triple bazată pe LLM + regex și validare ontologică
specifică domeniului legislativ.

## Prezentare generală

Sistemul procesează acte normative românești descărcate de pe
[legislatie.just.ro](https://legislatie.just.ro) și le transformă într-un
graf de cunoaștere interogabil semantic.

Caracteristici principale:
- **Corpus real** — 8 acte normative (legi, OUG, HG) stocate în `data/raw_laws/`
- **Extragere hibridă** — regex deterministic pentru referințe legislative +
  LLM (`gemma2:9b`) pentru relații semantice mai complexe
- **Vocabular închis** — 16 relații canonice; relațiile inventate de LLM sunt
  filtrate automat
- **Normalizare text** — NFC, corecție diacritice, eliminare antet
- **Segmentare articole** — algoritm LIS pentru filtrarea titlurilor de articole
  citate în corpul altui articol
- **Bază de cunoaștere vectorială** — ChromaDB cu embeddings `nomic-embed-text`
- **Raționament ontologic** — verificare proprietăți funcționale, asimetrice,
  ireflexive și constrângeri de domeniu

---

## Structura proiectului

```
Code/
├── data/
│   └── raw_laws/               # Acte normative reale (.txt + .meta.json)
│       └── SOURCES.md          # Bibliografie completă a corpusului
│
├── scripts/
│   ├── generate_meta.py        # Generare automată fișiere .meta.json
│   └── validate_pipeline.py    # Validare metrici + ontologie (Phase G)
│
├── src/                        # Pachet Python principal
│   ├── config.py               # Configurare modele și constante ontologie
│   ├── text_normalizer.py      # NFC + diacritice + eliminare antet
│   ├── law_loader.py           # Încărcare LawRecord = (law_id, text, meta)
│   ├── article_splitter.py     # Segmentare articole (LIS anti-poluare)
│   ├── relation_vocabulary.py  # 16 relații canonice + sinonime + prompt
│   ├── cross_reference_extractor.py  # Regex pre-pass referințe legislative
│   ├── llm_handler.py          # Prompt pipe-format + parser + vocab filter
│   ├── knowledge_base.py       # Stocare/interogare triple + ChromaDB
│   ├── graph_builder.py        # Construcție graf NetworkX
│   ├── ontology.py             # Raționament ontologic (6 tipuri de axiome)
│   ├── eda.py                  # Analiză exploratorie și vizualizări
│   └── legislative_generator.py # Generator legi sintetice (fallback/test)
│
├── output/                     # Fișiere generate (excluse din git)
│   ├── legislative_triples.csv
│   ├── legislative_corpus.csv
│   ├── knowledge_base_export.csv
│   └── legislative_knowledge_db/  # Baza vectorială ChromaDB
│
├── main.py                     # Pipeline principal
├── requirements.txt
├── pyproject.toml              # Configurare Ruff
└── README.md
```

---

## Pornire rapidă

### Prerequisite

- Python 3.11+
- [Ollama](https://ollama.ai) instalat și pornit (`ollama serve`)
- Modelele descărcate:
  ```bash
  ollama pull gemma2:9b
  ollama pull nomic-embed-text
  ```

### Instalare

```bash
git clone <repo>
cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Rulare pipeline

```bash
# Prima rulare — sau după ștergerea bazei vectoriale
python3 main.py

# Rulare rapidă (smoke test, primele 2 legi, 3 articole fiecare)
# Editează main.py: MAX_LAWS=2, MAX_ARTICLES_PER_LAW=3
python3 main.py
```

### Validare rezultate

```bash
python3 scripts/validate_pipeline.py

# Spot-check pe alte legi
python3 scripts/validate_pipeline.py --laws lege_24_2000 oug_13_2026
```

---

## Modulele principale

### `src/text_normalizer.py`
```python
from src.text_normalizer import normalize_ro

text = normalize_ro(raw_text)  # NFC + cedile → virgule + eliminare antet
```

### `src/law_loader.py`
```python
from src import load_real_laws

laws = load_real_laws("data/raw_laws")
# → [(law_id, normalized_text, metadata_dict), ...]
```

### `src/article_splitter.py`
```python
from src import split_into_articles

articles = split_into_articles(text)
# → [{"article_number": "1", "header": "Articolul 1", "text": "..."}, ...]
```

### `src/cross_reference_extractor.py`
Pre-pas regex determinist — detectează citări de tipul
`"modifică Legea nr. 53/2003"` fără a apela LLM-ul.

```python
from src import extract_cross_references

df = extract_cross_references(article_text, current_law_id="oug_13_2026")
# → DataFrame(head, relation, tail)
```

### `src/relation_vocabulary.py`
Vocabular închis cu 16 relații canonice. Folosit în prompt, parser și
validare:

| Relație | Semnificație |
|---|---|
| `emis_de` | cine a emis actul |
| `promulgat_de` | cine a promulgat |
| `publicat_în` | unde a fost publicat |
| `modifică` | ce act/articol modifică |
| `completează` | ce act completează |
| `abroga` | ce act abrogă |
| `introduce` | ce articol nou introduce |
| `republică` | ce act republică |
| `aprobă` | ce act aprobă |
| `face_referire_la` | trimitere (citare) |
| `intră_în_vigoare` | data/condiția de intrare în vigoare |
| `are_sediul_în` | unde are sediul o entitate |
| `responsabil_pentru` | responsabilitate |
| `colaborează_cu` | colaborare |
| `se_aplică` | domeniu de aplicare |
| `transpune` | transpunere directivă/regulament UE |

### `src/ontology.py`
```python
from src import LegislativeOntologyReasoner

reasoner = LegislativeOntologyReasoner(triples_df)
reasoner.run_all_tests()
# Verifică: funcționale, asimetrice, ireflexive, simetrice, domeniu, tranzitive
```

---

## Corpus legislativ

8 acte normative din `data/raw_laws/` (detalii complete în
[data/raw_laws/SOURCES.md](data/raw_laws/SOURCES.md)):

| `law_id` | Act | An |
|---|---|---|
| `lege_53_2003` | Codul muncii (republicare) | 2003/2011 |
| `lege_190_2018` | Lege GDPR Romania (aplicare Reg. UE 2016/679) | 2018 |
| `lege_24_2000` | Normele de tehnică legislativă | 2000/2004 |
| `oug_156_2024` | Măsuri fiscal-bugetare 2025 | 2024 |
| `oug_13_2026` | Modificări fiscal-bugetare | 2026 |
| `oug_5_2026` | Modificări instituții de credit | 2026 |
| `hg_214_2026` | Actualizare valori inventar SPP | 2026 |
| `hg_24_2026` | Eliberare din funcție subprefect | 2026 |

---

## Fișiere de ieșire

| Fișier | Conținut |
|---|---|
| `output/legislative_triples.csv` | Triple extrase (head, relation, tail, law_id, article_number) |
| `output/legislative_corpus.csv` | Corpusul de legi indexate |
| `output/knowledge_base_export.csv` | Export baza de cunoaștere |
| `output/legislative_knowledge_db/` | Baza vectorială ChromaDB persistentă |

---

## Configurare

Setările principale se află în `src/config.py`:

```python
LLM_MODEL = "gemma2:9b"
EMBEDDING_MODEL = "nomic-embed-text"

LEGISLATIVE_FUNCTIONAL_RELATIONS = ["emis_de", "promulgat_de"]
LEGISLATIVE_ASYMMETRIC_RELATIONS = ["modifică", "abroga"]
LEGISLATIVE_IRREFLEXIVE_RELATIONS = ["modifică", "promulgat_de", "emis_de"]
```

Comutatoare din `main.py`:

```python
USE_REAL_LAWS = True        # False → generator sintetic (pentru teste)
MAX_LAWS = None             # None → toate legile din data/raw_laws/
MAX_ARTICLES_PER_LAW = None # None → toate articolele
```

---

## Formatare cod

```bash
ruff format .          # formatare
ruff check --fix .     # linting cu auto-fix
pre-commit run --all-files
```

---

## Licență

MIT — vezi [LICENSE](LICENSE).

Textele legislative sunt proprietatea publică a statului român și sunt
disponibile la [legislatie.just.ro](https://legislatie.just.ro).
