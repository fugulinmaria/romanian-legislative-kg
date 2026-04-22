# Romanian Legislative Knowledge Graph Analysis

A modular framework for generating, extracting, storing, and reasoning over Romanian legislative knowledge graphs using LLM-powered triple extraction and ontological validation.

## 🎯 Project Overview

This system focuses on Romanian legislative text analysis with:
- **Synthetic law generation** - Create realistic Romanian legislative documents
- **LLM-powered extraction** - Extract knowledge triples from legislative texts
- **Knowledge base management** - Store and query legislative knowledge
- **Ontological reasoning** - Validate legislative knowledge using domain-specific axioms
- **Graph visualization** - Visualize relationships between laws and entities

## 📁 Project Structure

```
Code/
├── src/                          # Core package modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Legislative configuration
│   ├── data_loader.py           # Triple data utilities
│   ├── legislative_generator.py # Romanian law generation
│   ├── graph_builder.py         # Graph construction
│   ├── eda.py                   # Exploratory analysis
│   ├── ontology.py              # Legislative reasoning
│   ├── llm_handler.py           # LLM integration
│   └── knowledge_base.py        # Knowledge management
│
├── output/                      # All generated files
│   ├── legislative_triples.csv  # Extracted knowledge
│   ├── legislative_corpus.csv   # Law texts
│   ├── *.png                    # Visualizations
│   └── legislative_knowledge_db/# Vector database
│
├── .venv/                       # Virtual environment
├── main.py                      # Main analysis pipeline
├── requirements.txt             # Python dependencies
└── main_old_yago.py            # Archived YAGO analysis
```

## 🚀 Quick Start

### Prerequisites
✅ Already installed in this environment:
- Python 3.14.0
- Virtual environment (`.venv/`) with all dependencies:
  - `langchain-ollama`, `langchain-chroma`, `chromadb`
  - `pandas`, `matplotlib`, `networkx`
- Ollama with required models:
  - `gemma2:2b` (LLM for triple extraction)
  - `nomic-embed-text` (embeddings model)

### Running the Application

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Ensure Ollama is running:**
   ```bash
   # Check if Ollama service is running
   ollama list
   
   # If not running, start it:
   # ollama serve
   ```

3. **Run the main pipeline:**
   ```bash
   python3 main.py
   ```

### What the Pipeline Does:

1. **Initialize** - Load LLM models, embeddings, and vector store
2. **Generate** - Create diverse Romanian legislative texts
3. **Extract** - Use LLM to extract knowledge triples
4. **Build** - Construct knowledge graph from triples
5. **Analyze** - Perform EDA and create visualizations
6. **Reason** - Apply ontological validation rules
7. **Query** - Demonstrate knowledge base queries
7.5. **Semantic Search** - Find similar laws using vector embeddings
8. **Save** - Persist knowledge base and vector store to disk

## 📦 Module Overview

### `legislative_generator.py`
Generates realistic Romanian legislative texts:
- Simple laws
- Complex laws with chapters
- Amendment laws
- Emergency ordinances

```python
from src import RomanianLegislativeGenerator

generator = RomanianLegislativeGenerator()
law_text = generator.generate_complex_law(topic='digitalizarea administrativă')
```

### `knowledge_base.py`
Manages legislative knowledge:
- Store/retrieve triples
- Query by entity or relation
- Find modification chains
- Export knowledge base
- Semantic vector search (via Chroma)

```python
from src import LegislativeKnowledgeBase

kb = LegislativeKnowledgeBase()
kb.add_triples(triples_df)
modified_laws = kb.get_laws_modified_by('Legea nr. 450/2024')
```

### `ontology.py`
Legislative ontological reasoning with 6 axiom types:
- **Functional properties** - One emitter per law
- **Asymmetric properties** - No circular modifications
- **Irreflexive properties** - No self-reference
- **Symmetric properties** - Bidirectional collaboration
- **Domain constraints** - Valid entity types
- **Transitive analysis** - Modification chains

```python
from src import LegislativeOntologyReasoner

reasoner = LegislativeOntologyReasoner(triples_df)
reasoner.run_all_tests()
```

### `llm_handler.py`
LLM integration for triple extraction:
- Ollama model management
- Romanian text → triples
- Vector embeddings
- Chroma vector store

```python
from src import TripleExtractor, LLMHandler

llm = LLMHandler(temperature=0.1)
extractor = TripleExtractor(llm_handler=llm)
triples = extractor.extract_from_romanian_text(law_text)
```

## 🔧 Configuration

All settings in `src/config.py`:

```python
# Models
LLM_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "nomic-embed-text"

# Ontology Rules
LEGISLATIVE_FUNCTIONAL_RELATIONS = ['emis_de', 'promulgat_de']
LEGISLATIVE_ASYMMETRIC_RELATIONS = ['modifică', 'abroga']
LEGISLATIVE_IRREFLEXIVE_RELATIONS = ['modifică', 'promulgat_de', 'emis_de']

# Legislative Hierarchy
LAW_HIERARCHY = {
    'Constituție': 1,
    'Lege organică': 2,
    'Lege': 3,
    'Ordonanță de urgență': 4
}
```

## 📊 Output Files

All outputs saved to `output/`:

| File | Description |
|------|-------------|
| `legislative_triples.csv` | All extracted knowledge triples |
| `legislative_corpus.csv` | Generated law texts with IDs |
| `knowledge_base_export.csv` | Exported knowledge base |
| `legislative_relation_distribution.png` | Relation type analysis |
| `legislative_knowledge_graph.png` | Graph visualization |
| `legislative_knowledge_db/` | Chroma vector database with embeddings |

## 🎓 Key Features

### 1. Romanian Law Generation
```python
from src import generate_romanian_law

# Simple law
law = generate_romanian_law('simple', topic='educația națională')

# Complex law with chapters
law = generate_romanian_law('complex', topic='protecția mediului')
```

### 2. Knowledge Extraction
```python
from src import TripleExtractor, LLMHandler

llm = LLMHandler(temperature=0.1)
extractor = TripleExtractor(llm_handler=llm)

text = "Legea nr. 450/2024 este emisă de Parlamentul României."
triples = extractor.extract_from_romanian_text(text)
```

### 3. Ontological Validation
```python
from src import LegislativeOntologyReasoner

reasoner = LegislativeOntologyReasoner(triples_df)
reasoner.verify_functional_properties()
reasoner.verify_asymmetric_properties()
reasoner.verify_domain_constraints()
```

### 4. Knowledge Base Queries
```python
from src import LegislativeKnowledgeBase

kb = LegislativeKnowledgeBase()

# Find laws that modify a specific law
modifiers = kb.get_laws_modifying('Legea nr. 100/2018')

# Get who emitted a law
emitter = kb.get_law_emitter('Legea nr. 450/2024')

# Query by relation type
all

### 5. Semantic Vector Search
```python
from src import VectorStoreHandler, EmbeddingsHandler

embeddings = EmbeddingsHandler()
vector_store_handler = VectorStoreHandler(embeddings)
vector_store = vector_store_handler.get_store()

# Add law texts with embeddings
vector_store.add_texts(
    texts=[law_text],
    metadatas=[{"law_id": "law_123"}],
    ids=["law_123"]
)

# Semantic similarity search
results = vector_store.similarity_search("digitalizare și tehnologie", k=3)
for doc in results:
    print(f"Law: {doc.metadata['law_id']}")
    print(f"Text: {doc.page_content[:200]}...")
```_modifications = kb.query_by_relation('modifică')
```

## 🔄 Ontology Rules

### Functional Properties
- A law has ONE primary emitter
- A law is promulgated by ONE president
- A law is published in ONE official monitor

### Asymmetric Properties
- If� Installation (if setting up from scratch)

If you need to set up this project on a new machine:

1. **Install Python 3.8+**

2. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

3. **Pull required Ollama models:**
   ```bash
   ollama pull gemma2:2b
   ollama pull nomic-embed-text
   ```

4. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## 🧹 Code Quality

This project uses **ruff** for code formatting and linting.

### Manual Formatting
```bash
# Format all Python files
ruff format .

# Run linter with auto-fix
ruff check --fix .
```

### Pre-commit Hooks
Pre-commit hooks automatically format code before each commit:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Bypass hooks (not recommended)
git commit --no-verify
```

Configuration files:
- `.pre-commit-config.yaml` - Pre-commit hook setup
- `pyproject.toml` - Ruff configuration

## 📝 Notes

- All configuration in `src/config.py`
- Ollama must be running before executing the pipeline
- Knowledge base persists between runs in `output/`
- Vector store enables semantic search
- Generated visualizations saved to `output/` directory
- Old YAGO code archived in `main_old_yago.py` and `lege_old.py`

## ⚠️ Troubleshooting

- **Vector-based semantic search** for finding similar legislation
**"command not found: python"**
- Use `python3` instead of `python`

**"Connection refused" from Ollama**
- Start Ollama service: `ollama serve`
- Verify models are installed: `ollama list`

**Import errors**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Check installed packages: `pip list`
### Domain Constraints
- Only Parliament/Government can emit laws
- Only President can promulgate laws
- Only Official Monitor can publish laws

## 🎯 Research Focus

This system addresses:
- **Knowledge extraction** from Romanian legislative texts
- **Ontological reasoning** for legal knowledge validation
- **Knowledge graph construction** for legislative relationships
- **Semantic analysis** of legislative amendments and hierarchies

## 📝 Notes

- All configuration in `src/config.py`
- Requires Ollama with Romanian-capable LLM
- Knowledge base persists between runs
- Vector store enables semantic search
- Old YAGO code archived in `main_old_yago.py` and `lege_old.py`


