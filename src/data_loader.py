"""
Data loading utilities for Knowledge Graphs.
Handles loading triples from CSV files and external sources.
"""

import os

import pandas as pd

from .config import LEGISLATIVE_TRIPLES_PATH


class TripleDataLoader:
    """Generic loader for triple data from CSV files."""

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        if not os.path.exists(self.filepath):
            print(f"Warning: File {self.filepath} not found.")
            return pd.DataFrame(columns=["head", "relation", "tail"])

        print(f"Loading triples from {self.filepath}...")
        df = pd.read_csv(self.filepath)

        for col in ("head", "relation", "tail"):
            if col not in df.columns:
                raise ValueError("CSV must contain 'head', 'relation', and 'tail' columns")

        # Keep provenance columns (law_id, article_number) when present.
        return df

    def save(self, df):
        df.to_csv(self.filepath, index=False)
        print(f"Saved {len(df)} triples to {self.filepath}")


def load_legislative_triples():
    return TripleDataLoader(LEGISLATIVE_TRIPLES_PATH).load()


def save_legislative_triples(df):
    TripleDataLoader(LEGISLATIVE_TRIPLES_PATH).save(df)
