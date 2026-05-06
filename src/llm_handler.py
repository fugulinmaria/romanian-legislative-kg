"""
LLM Integration for Knowledge Graph Triple Extraction.
Manages Ollama models and provides utilities for extracting triples from text.
"""

import sys

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TEMPERATURE_EXTRACTION,
)


class LLMHandler:
    """Handles initialization and interaction with Ollama LLM models."""

    def __init__(self, model=LLM_MODEL, temperature=LLM_TEMPERATURE, verbose=True):
        if verbose:
            print(f"Initializing LLM: {model} (temperature={temperature})...")

        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature, timeout=120)

    def invoke(self, prompt_text):
        return self.llm.invoke(prompt_text)

    def create_chain(self, template):
        """Build a `prompt | llm` chain from a template string."""
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm

    def health_check(self, exit_on_failure: bool = True) -> bool:
        """Ping the Ollama server. Print guidance and exit on failure."""
        try:
            self.llm.invoke("ping")
            print(f"[OK] Ollama LLM '{self.model_name}' is reachable.")
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[FAIL] Cannot reach Ollama LLM '{self.model_name}': {exc}\n"
                f"  Start the server:    ollama serve\n"
                f"  Pull the model:      ollama pull {self.model_name}",
                file=sys.stderr,
            )
            if exit_on_failure:
                sys.exit(1)
            return False


class EmbeddingsHandler:
    """Handles initialization and interaction with embeddings models."""

    def __init__(self, model=EMBEDDING_MODEL, verbose=True):
        if verbose:
            print(f"Initializing embeddings model: {model}...")

        self.model_name = model
        self.embeddings = OllamaEmbeddings(model=model)

    def get_embeddings(self):
        return self.embeddings

    def health_check(self, exit_on_failure: bool = True) -> bool:
        """Ping the embeddings model. Print guidance and exit on failure."""
        try:
            self.embeddings.embed_query("ping")
            print(f"[OK] Ollama embeddings '{self.model_name}' is reachable.")
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[FAIL] Cannot reach Ollama embeddings '{self.model_name}': {exc}\n"
                f"  Start the server:    ollama serve\n"
                f"  Pull the model:      ollama pull {self.model_name}",
                file=sys.stderr,
            )
            if exit_on_failure:
                sys.exit(1)
            return False


class VectorStoreHandler:
    """Handles Chroma vector store initialization."""

    def __init__(self, embeddings_handler=None, verbose=True):
        if embeddings_handler is None:
            embeddings_handler = EmbeddingsHandler(verbose=verbose)

        if verbose:
            print(f"Initializing Chroma vector store at {CHROMA_DB_PATH}...")

        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings_handler.get_embeddings(),
            persist_directory=CHROMA_DB_PATH,
        )

    def get_store(self):
        return self.vector_store


class TripleExtractor:
    """Extracts knowledge graph triples from text using LLM."""

    def __init__(self, llm_handler=None):
        if llm_handler is None:
            self.llm_handler = LLMHandler(temperature=LLM_TEMPERATURE_EXTRACTION)
        else:
            self.llm_handler = llm_handler

    def extract_from_english_text(self, text):
        template = """Extract entities and relations as a list of triplets: [Entity1, Relation, Entity2].

Text: {text}

Triplets:"""

        chain = self.llm_handler.create_chain(template)
        return chain.invoke({"text": text})

    MAX_LLM_CHARS = 4000
    LLM_OVERLAP = 200

    @staticmethod
    def _chunk_for_llm(
        text: str, max_chars: int = MAX_LLM_CHARS, overlap: int = LLM_OVERLAP
    ) -> list[str]:
        """Split text into windows for LLM processing, with optional overlap."""

        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split("\n\n")
        windows: list[str] = []
        buf = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buf:
                    windows.append(buf)
                    buf = ""
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    windows.append(para[start:end])
                    if end == len(para):
                        break
                    start = end - overlap
                continue

            candidate = f"{buf}\n\n{para}" if buf else para
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    windows.append(buf)
                buf = para
        if buf:
            windows.append(buf)
        return windows

    def extract_from_romanian_text(self, text):
        """Extract canonical-vocabulary triples from a Romanian article."""
        from .relation_vocabulary import prompt_vocabulary_block

        windows = self._chunk_for_llm(text)
        if len(windows) > 1:
            print(f"  [INFO] Article split into {len(windows)} LLM windows " f"({len(text)} chars)")

        template = """Ești un sistem expert de extragere a informațiilor din texte legislative românești.

SARCINĂ: Extrage TOATE relațiile din textul de mai jos.

REGULI STRICTE:
1. Folosește DOAR relațiile din lista de mai jos. Dacă un fapt nu se încadrează în această listă, IGNORĂ-L.
2. Format de ieșire OBLIGATORIU: o tripletă pe linie, separată cu pipe (|), între paranteze pătrate:
   [Subiect | relație | Obiect]
3. NU folosi virgule ca separator. Subiectul și obiectul pot conține virgule normale.
4. Nu adăuga explicații, doar tripletele.

RELAȚII PERMISE:
{vocab}

EXEMPLE:
Text: "Legea nr. 100/2020 modifică Legea nr. 50/2015. A fost publicată în Monitorul Oficial nr. 200/2020."
Triplete:
[Legea nr. 100/2020 | modifică | Legea nr. 50/2015]
[Legea nr. 100/2020 | publicat_în | Monitorul Oficial nr. 200/2020]

Text: "Articolul 5 din Legea nr. 53/2003 - Codul muncii se modifică și va avea următorul cuprins: ..."
Triplete:
[Articolul 5 | modifică | Legea nr. 53/2003 - Codul muncii]

Acum extrage tripletele din acest text:

Text: {text}

Triplete:"""

        chain = self.llm_handler.create_chain(template)
        vocab = prompt_vocabulary_block()

        all_triples: list[list[str]] = []
        for i, window in enumerate(windows, 1):
            response = chain.invoke({"text": window, "vocab": vocab})
            if len(windows) > 1:
                print(
                    f"    [window {i}/{len(windows)}] response preview: "
                    f"{response[:120].strip()}..."
                )
            else:
                print(f"LLM Response preview: {response[:300]}...")

            triples = self._parse_triple_response(response)
            if not triples and len(windows) == 1:
                print(f"[WARN] No triples parsed from response. Full response:\n{response}")
            all_triples.extend(triples)

        df = pd.DataFrame(all_triples, columns=["head", "relation", "tail"])
        if not df.empty:
            df = df.drop_duplicates(subset=["head", "relation", "tail"], ignore_index=True)
        return df

    def _parse_triple_response(self, response):
        """Parse LLM response into structured triples.

        Accepts pipe-delimited format ``[head | relation | tail]`` (preferred)
        and falls back to comma-split for legacy responses. Filters relations
        through the canonical vocabulary; out-of-vocabulary triples are dropped.
        """
        from .relation_vocabulary import normalize_relation

        triples = []
        kept = 0
        dropped_unknown_rel = 0

        for raw_line in response.strip().split("\n"):
            line = raw_line.strip()
            if not line or line.lower().startswith(
                ("text:", "triplete:", "exemplu:", "format:", "reguli")
            ):
                continue
            if "[" not in line or "]" not in line:
                continue

            try:
                start_idx = line.index("[")
                end_idx = line.rindex("]")
                inner = line[start_idx + 1 : end_idx].strip()
                inner = inner.replace('"', "").replace("'", "")

                # Prefer pipe split (new format)
                if "|" in inner:
                    parts = [p.strip() for p in inner.split("|")]
                else:
                    # Legacy comma split — only safe when exactly 2 commas
                    parts = [p.strip() for p in inner.split(",")]
                    if len(parts) > 3:
                        # Re-join everything past the relation as the tail
                        parts = [parts[0], parts[1], ",".join(parts[2:]).strip()]

                if len(parts) < 3:
                    continue

                head, relation, tail = parts[0], parts[1], parts[2]
                if not (head and relation and tail):
                    continue

                canonical = normalize_relation(relation)
                if canonical is None:
                    dropped_unknown_rel += 1
                    continue

                triples.append([head, canonical, tail])
                kept += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] Failed to parse line: '{line[:80]}...' - {e}")
                continue

        print(f"  [OK] Parsed {kept} triples (dropped {dropped_unknown_rel} with unknown relation)")
        return triples


class KnowledgeGraphEvaluator:
    """Evaluates LLM extraction performance on knowledge graph data."""

    def __init__(self, llm_handler=None):
        if llm_handler is None:
            self.llm_handler = LLMHandler()
        else:
            self.llm_handler = llm_handler

        self.extractor = TripleExtractor(llm_handler=self.llm_handler)

    def evaluate_on_samples(self, df, sample_size=3):
        """Run the extractor on `sample_size` random rows and print results."""
        print("\n" + "=" * 60)
        print(f" LLM EXTRACTION EVALUATION ({sample_size} samples)")
        print("=" * 60)

        samples = df.sample(n=min(sample_size, len(df)))
        results = []

        for idx, row in samples.iterrows():
            text = f"{row['head']} {row['relation']} {row['tail']}"
            prediction = self.extractor.extract_from_english_text(text)

            print(f"\n--- Sample {len(results) + 1} ---")
            print(f"Input: {text}")
            print(f"Ground Truth: [{row['head']}, {row['relation']}, {row['tail']}]")
            print(f"LLM Output: {prediction.strip()}")

            results.append(
                {
                    "input": text,
                    "ground_truth": [row["head"], row["relation"], row["tail"]],
                    "prediction": prediction.strip(),
                }
            )

        return results


# Convenience functions for initialization
def init_llm_models(verbose=True):
    """Initialize LLM, embeddings and vector store; return them in a dict."""
    if verbose:
        print("Initializing local models (this may take a moment)...")

    embeddings_handler = EmbeddingsHandler(verbose=verbose)
    llm_handler = LLMHandler(verbose=verbose)
    vector_store_handler = VectorStoreHandler(
        embeddings_handler=embeddings_handler, verbose=verbose
    )

    return {
        "llm": llm_handler,
        "embeddings": embeddings_handler,
        "vector_store": vector_store_handler.get_store(),
    }
