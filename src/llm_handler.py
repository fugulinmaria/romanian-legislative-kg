"""
LLM Integration for Knowledge Graph Triple Extraction.
Manages Ollama models and provides utilities for extracting triples from text.
"""

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
        """
        Initialize LLM handler.

        Args:
            model (str): Ollama model name
            temperature (float): Temperature for generation
            verbose (bool): Whether to print initialization messages
        """
        if verbose:
            print(f"Initializing LLM: {model} (temperature={temperature})...")

        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature, timeout=120)

    def invoke(self, prompt_text):
        """
        Invoke the LLM with a prompt.

        Args:
            prompt_text (str): The prompt to send to the LLM

        Returns:
            str: LLM response
        """
        return self.llm.invoke(prompt_text)

    def create_chain(self, template):
        """
        Create a LangChain chain with a prompt template.

        Args:
            template (str): Prompt template string

        Returns:
            Chain: LangChain chain object
        """
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm


class EmbeddingsHandler:
    """Handles initialization and interaction with embeddings models."""

    def __init__(self, model=EMBEDDING_MODEL, verbose=True):
        """
        Initialize embeddings handler.

        Args:
            model (str): Ollama embeddings model name
            verbose (bool): Whether to print initialization messages
        """
        if verbose:
            print(f"Initializing embeddings model: {model}...")

        self.model_name = model
        self.embeddings = OllamaEmbeddings(model=model)

    def get_embeddings(self):
        """Get the embeddings function."""
        return self.embeddings


class VectorStoreHandler:
    """Handles Chroma vector store initialization."""

    def __init__(self, embeddings_handler=None, verbose=True):
        """
        Initialize vector store handler.

        Args:
            embeddings_handler (EmbeddingsHandler, optional): Embeddings handler
            verbose (bool): Whether to print initialization messages
        """
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
        """Get the vector store."""
        return self.vector_store


class TripleExtractor:
    """Extracts knowledge graph triples from text using LLM."""

    def __init__(self, llm_handler=None):
        """
        Initialize triple extractor.

        Args:
            llm_handler (LLMHandler, optional): LLM handler instance
        """
        if llm_handler is None:
            self.llm_handler = LLMHandler(temperature=LLM_TEMPERATURE_EXTRACTION)
        else:
            self.llm_handler = llm_handler

    def extract_from_english_text(self, text):
        """
        Extract triples from English text.

        Args:
            text (str): Input text

        Returns:
            str: Raw LLM response with extracted triples
        """
        template = """Extract entities and relations as a list of triplets: [Entity1, Relation, Entity2].

Text: {text}

Triplets:"""

        chain = self.llm_handler.create_chain(template)
        return chain.invoke({"text": text})

    def extract_from_romanian_text(self, text):
        """
        Extract triples from Romanian legislative text.

        Args:
            text (str): Input Romanian text

        Returns:
            pd.DataFrame: DataFrame with columns ['head', 'relation', 'tail']
        """
        from .relation_vocabulary import prompt_vocabulary_block

        # Truncate long articles — the LLM only needs the first ~2 000 chars to
        # find canonical relations; the regex pre-pass already handled citations.
        MAX_LLM_CHARS = 2000
        if len(text) > MAX_LLM_CHARS:
            text = text[:MAX_LLM_CHARS]

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
        response = chain.invoke({"text": text, "vocab": prompt_vocabulary_block()})

        print(f"LLM Response preview: {response[:300]}...")

        triples = self._parse_triple_response(response)

        if not triples:
            print(f"[WARN] No triples parsed from response. Full response:\n{response}")

        return pd.DataFrame(triples, columns=["head", "relation", "tail"])

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
        """
        Initialize evaluator.

        Args:
            llm_handler (LLMHandler, optional): LLM handler instance
        """
        if llm_handler is None:
            self.llm_handler = LLMHandler()
        else:
            self.llm_handler = llm_handler

        self.extractor = TripleExtractor(llm_handler=self.llm_handler)

    def evaluate_on_samples(self, df, sample_size=3):
        """
        Evaluate LLM extraction on random samples from dataset.

        Args:
            df (pd.DataFrame): Dataset with triples
            sample_size (int): Number of samples to test

        Returns:
            list: List of (input, ground_truth, prediction) tuples
        """
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
    """
    Initialize all LLM models and services.

    Args:
        verbose (bool): Whether to print initialization messages

    Returns:
        dict: Dictionary with 'llm', 'embeddings', and 'vector_store'
    """
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
