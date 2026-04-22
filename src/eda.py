"""
Exploratory Data Analysis and Visualization for Knowledge Graphs.
Provides comprehensive analysis and plotting functions.
"""

import matplotlib.pyplot as plt
import networkx as nx

from .config import PLOT_STYLE, get_plot_path


class KnowledgeGraphEDA:
    """Performs exploratory data analysis on knowledge graphs."""

    def __init__(self, df, graph=None):
        """
        Initialize EDA analyzer.

        Args:
            df (pd.DataFrame): DataFrame with columns ['head', 'relation', 'tail']
            graph (nx.MultiDiGraph, optional): NetworkX graph
        """
        self.df = df
        self.graph = graph

    def print_basic_stats(self):
        """Print basic dataset statistics."""
        print("\n" + "=" * 60)
        print(" DATASET STATISTICS")
        print("=" * 60)

        total_triples = len(self.df)
        unique_heads = self.df["head"].nunique()
        unique_tails = self.df["tail"].nunique()
        unique_relations = self.df["relation"].nunique()

        print(f"Total Triples (Rows): {total_triples:,}")
        print(f"Unique Subject Entities (Heads): {unique_heads:,}")
        print(f"Unique Object Entities (Tails): {unique_tails:,}")
        print(f"Unique Relationship Types (Predicates): {unique_relations:,}")

        missing_data = self.df.isnull().sum().sum()
        print(f"Missing Values (Nulls): {missing_data}")

        num_duplicates = len(
            self.df[self.df.duplicated(subset=["head", "relation", "tail"], keep=False)]
        )
        if total_triples > 0:
            redundancy_rate = (num_duplicates / total_triples) * 100
            print(f"Duplicate Triples: {num_duplicates:,} ({redundancy_rate:.2f}%)")

    def print_graph_metrics(self):
        """Print graph topology metrics."""
        if self.graph is None:
            print("\nWarning: No graph provided. Skipping graph metrics.")
            return

        print("\n" + "=" * 60)
        print(" GRAPH TOPOLOGY METRICS")
        print("=" * 60)

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        reciprocity = nx.reciprocity(self.graph)

        print(f"Total Nodes (Entities): {nodes:,}")
        print(f"Total Edges (Triples): {edges:,}")
        print(f"Graph Density: {density * 100:.6f}%")
        print(f"Graph Reciprocity: {reciprocity * 100:.4f}%")

    def print_relational_distribution(self, top_n=10):
        """
        Print relationship type distribution.

        Args:
            top_n (int): Number of top/bottom relations to show
        """
        print("\n" + "=" * 60)
        print(" RELATIONAL DISTRIBUTION")
        print("=" * 60)

        relation_counts = self.df["relation"].value_counts()
        print(f"Total Unique Relations (Predicates): {len(relation_counts)}")

        print(f"\nTop {top_n} Most Frequent Relations:")
        print(relation_counts.head(top_n))

        print(f"\nBottom {min(top_n, len(relation_counts))} Least Frequent Relations:")
        print(relation_counts.tail(top_n))

    def print_top_entities(self, top_n=5):
        """
        Print most referenced entities.

        Args:
            top_n (int): Number of top entities to show
        """
        print(f"\nTop {top_n} Most Referenced Entities (As Subject):")
        print(self.df["head"].value_counts().head(top_n))

        print(f"\nTop {top_n} Most Referenced Entities (As Object):")
        print(self.df["tail"].value_counts().head(top_n))

    def plot_relational_distribution(self, top_n=15, filename="relation_distribution"):
        """
        Plot bar chart of relationship type distribution.

        Args:
            top_n (int): Number of top relations to plot
            filename (str): Output filename (without extension)
        """
        print(f"\nPlotting relational distribution (Top {top_n})...")

        relation_counts = self.df["relation"].value_counts().head(top_n)

        plt.figure(figsize=PLOT_STYLE["figure_size_large"])
        relation_counts.plot(
            kind="bar", color=PLOT_STYLE["color_secondary"], edgecolor=PLOT_STYLE["edge_color"]
        )
        plt.title(f"Top {top_n} Most Frequent Relationships")
        plt.xlabel("Relationship Type")
        plt.ylabel("Frequency (Number of Triples)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = get_plot_path(filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to '{output_path}'")

    def plot_degree_distribution(self, filename="degree_distribution"):
        """
        Plot node degree distribution.

        Args:
            filename (str): Output filename (without extension)
        """
        if self.graph is None:
            print("\nWarning: No graph provided. Cannot plot degree distribution.")
            return

        print("\nPlotting node degree distribution...")

        degrees = [d for n, d in self.graph.degree()]

        plt.figure(figsize=PLOT_STYLE["figure_size_medium"])
        plt.hist(
            degrees,
            bins=50,
            log=True,
            color=PLOT_STYLE["color_primary"],
            edgecolor=PLOT_STYLE["edge_color"],
        )
        plt.title("Node Degree Distribution (Log Scale)")
        plt.xlabel("Degree (Number of Connections per Node)")
        plt.ylabel("Frequency of Nodes (Log Scale)")
        plt.tight_layout()

        output_path = get_plot_path(filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to '{output_path}'")

    def plot_in_out_degree_distribution(self, filename="in_out_degree_distribution"):
        """
        Plot separate in-degree and out-degree distributions.

        Args:
            filename (str): Output filename (without extension)
        """
        if self.graph is None:
            print("\nWarning: No graph provided. Cannot plot degree distribution.")
            return

        print("\nPlotting in-degree vs out-degree distribution...")

        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]

        plt.figure(figsize=PLOT_STYLE["figure_size_split"])

        plt.subplot(1, 2, 1)
        plt.hist(
            in_degrees,
            bins=50,
            log=True,
            color=PLOT_STYLE["color_tertiary"],
            edgecolor=PLOT_STYLE["edge_color"],
        )
        plt.title("In-Degree Distribution (Log Scale)")
        plt.xlabel("In-Degree")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(
            out_degrees,
            bins=50,
            log=True,
            color=PLOT_STYLE["color_primary"],
            edgecolor=PLOT_STYLE["edge_color"],
        )
        plt.title("Out-Degree Distribution (Log Scale)")
        plt.xlabel("Out-Degree")

        plt.tight_layout()

        output_path = get_plot_path(filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to '{output_path}'")

    def plot_knowledge_graph_sample(self, max_nodes=50, filename="knowledge_graph_visualization"):
        """
        Plot a sample visualization of the knowledge graph.

        Args:
            max_nodes (int): Maximum number of nodes to visualize
            filename (str): Output filename (without extension)
        """
        if self.graph is None:
            print("\nWarning: No graph provided. Cannot visualize graph.")
            return

        print(f"\nGenerating graph visualization (max {max_nodes} nodes)...")

        # Create subgraph if too large
        if self.graph.number_of_nodes() > max_nodes:
            nodes = list(self.graph.nodes())[:max_nodes]
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_color=PLOT_STYLE["color_primary"],
            font_size=8,
            node_size=2000,
            arrowsize=15,
            edge_color="gray",
            alpha=0.7,
        )

        edge_labels = {(u, v): d["relation"] for u, v, d in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_color="red", font_size=6
        )

        plt.title(f"Knowledge Graph Sample ({subgraph.number_of_nodes()} nodes)")
        plt.axis("off")
        plt.tight_layout()

        output_path = get_plot_path(filename)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved visualization to '{output_path}'")

    def run_full_eda(self):
        """Run complete EDA analysis with all visualizations."""
        self.print_basic_stats()
        self.print_graph_metrics()
        self.print_relational_distribution()
        self.print_top_entities()
        self.plot_relational_distribution()
        self.plot_degree_distribution()
        self.plot_in_out_degree_distribution()
