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
        self.llm = OllamaLLM(model=model, temperature=temperature)

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
        print("\nExtracting triples from Romanian text using LLM...")

        template = """Ești un sistem expert de extragere a informațiilor din texte legislative românești.

SARCINĂ: Extrage TOATE relațiile din textul legislativ de mai jos. Pentru fiecare articol, identifică toate faptele.

RELAȚII DE CĂUTAT:
- emis_de: cine a emis legea (Parlamentul, Guvernul)
- modifică: ce legi modifică
- abroga: ce articole sau legi abrogă
- completează: ce legi completează
- promulgat_de: cine a promulgat (Președintele)
- publicat_în: unde a fost publicată (Monitorul Oficial)
- are_sediul_în: unde are sediul o entitate
- responsabil_pentru: cine este responsabil
- colaborează_cu: cine colaborează cu cine
- intră_în_vigoare: când intră în vigoare

FORMAT DE IEȘIRE: O tripletă pe linie în format: [Subiect, relație, Obiect]

EXEMPLU:
Text: "Legea nr. 100/2020 este emisă de Parlamentul României. Această lege modifică Legea nr. 50/2015."
Triplete:
[Legea nr. 100/2020, emis_de, Parlamentul României]
[Legea nr. 100/2020, modifică, Legea nr. 50/2015]

Acum extrage TOATE tripletele din acest text:

Text: {text}

Triplete:"""

        chain = self.llm_handler.create_chain(template)
        response = chain.invoke({"text": text})

        print(f"LLM Response preview: {response[:300]}...")

        triples = self._parse_triple_response(response)

        if not triples:
            print(f"[WARN] No triples parsed from response. Full response:\n{response}")

        return pd.DataFrame(triples, columns=["head", "relation", "tail"])

    def _parse_triple_response(self, response):
        """
        Parse LLM response into structured triples.

        Args:
            response (str): Raw LLM response

        Returns:
            list: List of [head, relation, tail] triples
        """
        triples = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            if not line or line.lower().startswith(("text:", "triplete:", "exemplu:", "format:")):
                continue

            if "[" in line and "]" in line:
                try:
                    start_idx = line.index("[")
                    end_idx = line.rindex("]")
                    triple_str = line[start_idx + 1 : end_idx]

                    triple_str = triple_str.replace('"', "").replace("'", "")

                    parts = [p.strip() for p in triple_str.split(",")]

                    if len(parts) >= 3:
                        head = parts[0].strip()
                        relation = parts[1].strip()
                        tail = ",".join(parts[2:]).strip()

                        if head and relation and tail:
                            triples.append([head, relation, tail])
                except Exception as e:
                    print(f"  [WARN] Failed to parse line: '{line[:80]}...' - Error: {e}")
                    continue

        print(f"  [OK] Parsed {len(triples)} triples from LLM response")
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
