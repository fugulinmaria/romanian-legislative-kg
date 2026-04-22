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
        """
        Initialize data loader.

        Args:
            filepath (str): Path to CSV file with triples
        """
        self.filepath = filepath

    def load(self):
        """
        Load triples from CSV file.

        Returns:
            pd.DataFrame: DataFrame with columns ['head', 'relation', 'tail']
        """
        if not os.path.exists(self.filepath):
            print(f"Warning: File {self.filepath} not found.")
            return pd.DataFrame(columns=["head", "relation", "tail"])

        print(f"Loading triples from {self.filepath}...")
        df = pd.read_csv(self.filepath)

        if "head" not in df.columns or "relation" not in df.columns or "tail" not in df.columns:
            raise ValueError("CSV must contain 'head', 'relation', and 'tail' columns")

        return df[["head", "relation", "tail"]]

    def save(self, df):
        """
        Save triples to CSV file.

        Args:
            df (pd.DataFrame): DataFrame with triples
        """
        df[["head", "relation", "tail"]].to_csv(self.filepath, index=False)
        print(f"Saved {len(df)} triples to {self.filepath}")


def load_legislative_triples():
    """
    Load legislative triples from the knowledge base.

    Returns:
        pd.DataFrame: Legislative triples
    """
    loader = TripleDataLoader(LEGISLATIVE_TRIPLES_PATH)
    return loader.load()


def save_legislative_triples(df):
    """
    Save legislative triples to the knowledge base.

    Args:
        df (pd.DataFrame): Legislative triples DataFrame
    """
    loader = TripleDataLoader(LEGISLATIVE_TRIPLES_PATH)
    loader.save(df)
