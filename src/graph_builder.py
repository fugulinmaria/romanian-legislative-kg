"""
Graph building and operations for Knowledge Graphs.
Provides utilities for creating and analyzing NetworkX graphs from triple data.
"""

import re

import networkx as nx

from .law_id_resolver import get_meta

_LAW_ID_RE = re.compile(r"^[a-z]+(?:_\d+){2,}$")


class KnowledgeGraphBuilder:
    """Builds and manages NetworkX graphs from triple data."""

    def __init__(self):
        self.graph = None

    def build_from_dataframe(self, df, verbose=True):
        """Build a MultiDiGraph from a triples DataFrame.

        If `law_id` / `article_number` columns are present, they are attached
        as edge attributes for downstream provenance.
        """
        if verbose:
            print(f"Building NetworkX graph from {len(df):,} triples...")

        self.graph = nx.MultiDiGraph()

        edge_attrs = ["relation"]
        for col in ("law_id", "article_number", "source_method", "confidence"):
            if col in df.columns:
                edge_attrs.append(col)

        # Cast to str so node identities are stable.
        edge_df = df.assign(
            head=df["head"].astype(str),
            tail=df["tail"].astype(str),
        )

        self.graph = nx.from_pandas_edgelist(
            edge_df,
            source="head",
            target="tail",
            edge_attr=edge_attrs,
            create_using=nx.MultiDiGraph,
        )

        self._annotate_law_nodes()

        if verbose:
            print(
                f"Graph built: {self.graph.number_of_nodes():,} nodes, "
                f"{self.graph.number_of_edges():,} edges"
            )

        return self.graph

    def get_graph_stats(self):
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_from_dataframe first.")

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "reciprocity": nx.reciprocity(self.graph),
        }

    def get_degree_distributions(self):
        """Return (in_degrees, out_degrees, total_degrees)."""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_from_dataframe first.")

        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        total_degrees = [d for n, d in self.graph.degree()]

        return in_degrees, out_degrees, total_degrees

    def _annotate_law_nodes(self):
        """Attach node_type='law' and metadata to law_id nodes."""
        for node in self.graph.nodes:
            if not isinstance(node, str) or not _LAW_ID_RE.match(node):
                continue
            self.graph.nodes[node]["node_type"] = "law"
            meta = get_meta(node)
            if meta:
                for k in ("tip_act", "numar", "an", "titlu", "in_force"):
                    if k in meta:
                        self.graph.nodes[node][k] = meta[k]
            else:
                self.graph.nodes[node]["external"] = True


def build_graph_from_triples(df, verbose=True):
    builder = KnowledgeGraphBuilder()
    return builder.build_from_dataframe(df, verbose=verbose)
