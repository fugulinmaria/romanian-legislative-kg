"""
Legislative Knowledge Base Management
Stores, retrieves, and queries Romanian legislative knowledge graphs.
"""

import os

import pandas as pd

from .config import LEGISLATIVE_CORPUS_PATH, OUTPUT_DIR
from .data_loader import load_legislative_triples, save_legislative_triples


class LegislativeKnowledgeBase:
    """Manages storage and retrieval of legislative knowledge."""

    def __init__(self):
        """Initialize the knowledge base."""
        self.triples_df = pd.DataFrame(columns=["head", "relation", "tail"])
        self.corpus = []  # Stores (law_id, law_text) tuples
        self.load()

    def load(self):
        """Load existing knowledge base from disk."""
        try:
            self.triples_df = load_legislative_triples()
            if not self.triples_df.empty:
                print(f"Loaded {len(self.triples_df)} triples from knowledge base.")
        except Exception as e:
            print(f"No existing knowledge base found or error loading: {e}")
            self.triples_df = pd.DataFrame(columns=["head", "relation", "tail"])

    def add_triples(self, triples_df):
        """
        Add new triples to the knowledge base.

        Args:
            triples_df (pd.DataFrame): DataFrame with columns ['head', 'relation', 'tail']
        """
        if triples_df.empty:
            print("Warning: No triples to add.")
            return

        self.triples_df = pd.concat([self.triples_df, triples_df], ignore_index=True)

        initial_count = len(self.triples_df)
        self.triples_df = self.triples_df.drop_duplicates(subset=["head", "relation", "tail"])
        duplicates_removed = initial_count - len(self.triples_df)

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate triples.")

        print(f"Added triples. Total: {len(self.triples_df)}")

    def add_law_to_corpus(self, law_id, law_text):
        """
        Add a law to the corpus.

        Args:
            law_id (str): Unique identifier for the law
            law_text (str): Full text of the law
        """
        self.corpus.append((law_id, law_text))
        print(f"Added law '{law_id}' to corpus. Total laws: {len(self.corpus)}")

    def save(self):
        """Save knowledge base to disk."""
        save_legislative_triples(self.triples_df)

        if self.corpus:
            corpus_df = pd.DataFrame(self.corpus, columns=["law_id", "law_text"])
            corpus_df.to_csv(LEGISLATIVE_CORPUS_PATH, index=False)
            print(f"Saved {len(self.corpus)} laws to corpus.")

    def query_by_entity(self, entity, role="any"):
        """
        Query triples by entity.

        Args:
            entity (str): Entity to search for
            role (str): 'head', 'tail', or 'any'

        Returns:
            pd.DataFrame: Matching triples
        """
        if role == "head":
            return self.triples_df[self.triples_df["head"] == entity]
        elif role == "tail":
            return self.triples_df[self.triples_df["tail"] == entity]
        else:  # any
            return self.triples_df[
                (self.triples_df["head"] == entity) | (self.triples_df["tail"] == entity)
            ]

    def query_by_relation(self, relation):
        """
        Query triples by relation type.

        Args:
            relation (str): Relation to search for

        Returns:
            pd.DataFrame: Matching triples
        """
        return self.triples_df[self.triples_df["relation"] == relation]

    def get_laws_modified_by(self, law_id):
        """
        Get all laws modified by a specific law.

        Args:
            law_id (str): Law identifier

        Returns:
            List[str]: List of modified law IDs
        """
        result = self.triples_df[
            (self.triples_df["head"] == law_id) & (self.triples_df["relation"] == "modifică")
        ]
        return result["tail"].tolist()

    def get_laws_modifying(self, law_id):
        """
        Get all laws that modify a specific law.

        Args:
            law_id (str): Law identifier

        Returns:
            List[str]: List of modifying law IDs
        """
        result = self.triples_df[
            (self.triples_df["tail"] == law_id) & (self.triples_df["relation"] == "modifică")
        ]
        return result["head"].tolist()

    def get_law_emitter(self, law_id):
        """
        Get the emitter of a specific law.

        Args:
            law_id (str): Law identifier

        Returns:
            Optional[str]: Emitter entity or None
        """
        result = self.triples_df[
            (self.triples_df["head"] == law_id) & (self.triples_df["relation"] == "emis_de")
        ]
        return result["tail"].iloc[0] if not result.empty else None

    def get_entity_relations(self, entity):
        """
        Get all relations for an entity.

        Args:
            entity (str): Entity to analyze

        Returns:
            Dict: Dictionary with 'outgoing' and 'incoming' relations
        """
        outgoing = self.triples_df[self.triples_df["head"] == entity]
        incoming = self.triples_df[self.triples_df["tail"] == entity]

        return {
            "outgoing": outgoing[["relation", "tail"]].to_dict("records"),
            "incoming": incoming[["head", "relation"]].to_dict("records"),
        }

    def get_statistics(self):
        """
        Get knowledge base statistics.

        Returns:
            Dict: Statistics about the knowledge base
        """
        return {
            "total_triples": len(self.triples_df),
            "unique_entities": len(
                set(self.triples_df["head"].tolist() + self.triples_df["tail"].tolist())
            ),
            "unique_relations": self.triples_df["relation"].nunique(),
            "total_laws_in_corpus": len(self.corpus),
            "relation_distribution": self.triples_df["relation"].value_counts().to_dict(),
        }

    def export_to_csv(self, filepath=None):
        """
        Export triples to CSV.

        Args:
            filepath (str, optional): Custom export path
        """
        if filepath is None:
            filepath = os.path.join(OUTPUT_DIR, "knowledge_base_export.csv")

        self.triples_df.to_csv(filepath, index=False)
        print(f"Exported knowledge base to {filepath}")

    def clear(self):
        """Clear all data from the knowledge base."""
        self.triples_df = pd.DataFrame(columns=["head", "relation", "tail"])
        self.corpus = []
        print("Knowledge base cleared.")

    def get_all_triples(self):
        """
        Get all triples in the knowledge base.

        Returns:
            pd.DataFrame: All triples
        """
        return self.triples_df.copy()


# Convenience function
def create_knowledge_base():
    """
    Create and return a new knowledge base instance.

    Returns:
        LegislativeKnowledgeBase: Knowledge base instance
    """
    return LegislativeKnowledgeBase()
