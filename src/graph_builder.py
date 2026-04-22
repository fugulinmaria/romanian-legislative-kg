"""
Graph building and operations for Knowledge Graphs.
Provides utilities for creating and analyzing NetworkX graphs from triple data.
"""

import networkx as nx


class KnowledgeGraphBuilder:
    """Builds and manages NetworkX graphs from triple data."""

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = None

    def build_from_dataframe(self, df, verbose=True):
        """
        Build a NetworkX MultiDiGraph from a DataFrame of triples.

        Args:
            df (pd.DataFrame): DataFrame with columns ['head', 'relation', 'tail']
            verbose (bool): Whether to print progress messages

        Returns:
            nx.MultiDiGraph: The constructed knowledge graph
        """
        if verbose:
            print(f"Building NetworkX graph from {len(df):,} triples...")

        self.graph = nx.MultiDiGraph()

        for _, row in df.iterrows():
            self.graph.add_edge(str(row["head"]), str(row["tail"]), relation=str(row["relation"]))

        if verbose:
            print(
                f"Graph built: {self.graph.number_of_nodes():,} nodes, "
                f"{self.graph.number_of_edges():,} edges"
            )

        return self.graph

    def get_graph_stats(self):
        """
        Get basic statistics about the graph.

        Returns:
            dict: Dictionary containing graph statistics
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_from_dataframe first.")

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "reciprocity": nx.reciprocity(self.graph),
        }

    def get_degree_distributions(self):
        """
        Get in-degree and out-degree distributions.

        Returns:
            tuple: (in_degrees, out_degrees) as lists
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_from_dataframe first.")

        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        total_degrees = [d for n, d in self.graph.degree()]

        return in_degrees, out_degrees, total_degrees


# Convenience function
def build_graph_from_triples(df, verbose=True):
    """
    Convenience function to build a graph from triples DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['head', 'relation', 'tail']
        verbose (bool): Whether to print progress

    Returns:
        nx.MultiDiGraph: The constructed knowledge graph
    """
    builder = KnowledgeGraphBuilder()
    return builder.build_from_dataframe(df, verbose=verbose)
