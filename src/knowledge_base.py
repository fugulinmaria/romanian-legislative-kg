"""
Legislative Knowledge Base Management
Stores, retrieves, and queries Romanian legislative knowledge graphs.
"""

import os

import pandas as pd

from .config import LEGISLATIVE_CORPUS_PATH, LEGISLATIVE_TRIPLES_PATH, OUTPUT_DIR

TRIPLE_COLS = ["head", "relation", "tail", "law_id", "article_number"]


class LegislativeKnowledgeBase:
    """Manages storage and retrieval of legislative knowledge."""

    def __init__(self):
        self.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
        self.corpus = []  # (law_id, law_text)
        self.load()

    def load(self):
        try:
            if not os.path.exists(LEGISLATIVE_TRIPLES_PATH):
                print(f"Warning: File {LEGISLATIVE_TRIPLES_PATH} not found.")
                self.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
                return

            print(f"Loading triples from {LEGISLATIVE_TRIPLES_PATH}...")
            self.triples_df = pd.read_csv(LEGISLATIVE_TRIPLES_PATH)
            for col in ("head", "relation", "tail"):
                if col not in self.triples_df.columns:
                    raise ValueError("CSV must contain 'head', 'relation', and 'tail' columns")
            for col in ("law_id", "article_number"):
                if col not in self.triples_df.columns:
                    self.triples_df[col] = pd.NA
            if not self.triples_df.empty:
                print(f"Loaded {len(self.triples_df)} triples from knowledge base.")
        except Exception as e:
            print(f"No existing knowledge base found or error loading: {e}")
            self.triples_df = pd.DataFrame(columns=TRIPLE_COLS)

    def add_triples(self, triples_df):
        if triples_df.empty:
            print("Warning: No triples to add.")
            return

        incoming = triples_df.copy()
        for col in ("law_id", "article_number"):
            if col not in incoming.columns:
                incoming[col] = pd.NA
        incoming = incoming[TRIPLE_COLS]

        self.triples_df = pd.concat([self.triples_df, incoming], ignore_index=True)

        initial_count = len(self.triples_df)
        self.triples_df = self.triples_df.drop_duplicates(subset=["head", "relation", "tail"])
        duplicates_removed = initial_count - len(self.triples_df)

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate triples.")

        print(f"Added triples. Total: {len(self.triples_df)}")

    def add_law_to_corpus(self, law_id, law_text):
        self.corpus.append((law_id, law_text))
        print(f"Added law '{law_id}' to corpus. Total laws: {len(self.corpus)}")

    def save(self):
        self.triples_df.to_csv(LEGISLATIVE_TRIPLES_PATH, index=False)
        print(f"Saved {len(self.triples_df)} triples to {LEGISLATIVE_TRIPLES_PATH}")

        if self.corpus:
            corpus_df = pd.DataFrame(self.corpus, columns=["law_id", "law_text"])
            corpus_df.to_csv(LEGISLATIVE_CORPUS_PATH, index=False)
            print(f"Saved {len(self.corpus)} laws to corpus.")

    def query_by_entity(self, entity, role="any"):
        """role: 'head', 'tail', or 'any'."""
        if role == "head":
            return self.triples_df[self.triples_df["head"] == entity]
        elif role == "tail":
            return self.triples_df[self.triples_df["tail"] == entity]
        else:  # any
            return self.triples_df[
                (self.triples_df["head"] == entity) | (self.triples_df["tail"] == entity)
            ]

    def query_by_relation(self, relation):
        return self.triples_df[self.triples_df["relation"] == relation]

    def query_by_law(self, law_id):
        """All triples extracted from `law_id`."""
        return self.triples_df[self.triples_df["law_id"] == law_id]

    def query_by_article(self, law_id, article_number):
        """All triples extracted from a specific article."""
        return self.triples_df[
            (self.triples_df["law_id"] == law_id)
            & (self.triples_df["article_number"].astype(str) == str(article_number))
        ]

    def get_laws_modified_by(self, law_id):
        result = self.triples_df[
            (self.triples_df["head"] == law_id) & (self.triples_df["relation"] == "modifică")
        ]
        return result["tail"].tolist()

    def get_laws_modifying(self, law_id):
        result = self.triples_df[
            (self.triples_df["tail"] == law_id) & (self.triples_df["relation"] == "modifică")
        ]
        return result["head"].tolist()

    def get_law_emitter(self, law_id):
        result = self.triples_df[
            (self.triples_df["head"] == law_id) & (self.triples_df["relation"] == "emis_de")
        ]
        return result["tail"].iloc[0] if not result.empty else None

    def get_entity_relations(self, entity):
        """Return {'outgoing': [...], 'incoming': [...]} for an entity."""
        outgoing = self.triples_df[self.triples_df["head"] == entity]
        incoming = self.triples_df[self.triples_df["tail"] == entity]

        return {
            "outgoing": outgoing[["relation", "tail"]].to_dict("records"),
            "incoming": incoming[["head", "relation"]].to_dict("records"),
        }

    def get_statistics(self):
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
        if filepath is None:
            filepath = os.path.join(OUTPUT_DIR, "knowledge_base_export.csv")

        self.triples_df.to_csv(filepath, index=False)
        print(f"Exported knowledge base to {filepath}")

    def clear(self):
        self.triples_df = pd.DataFrame(columns=TRIPLE_COLS)
        self.corpus = []
        print("Knowledge base cleared.")

    def get_all_triples(self):
        return self.triples_df.copy()
