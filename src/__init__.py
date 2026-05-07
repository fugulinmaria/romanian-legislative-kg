"""
Romanian Legislative Knowledge Graph Analysis Package
Provides modular tools for generating, extracting, and reasoning
over Romanian legislative knowledge.
"""

from .article_splitter import split_into_articles
from .config import *  # noqa: F403
from .cross_reference_extractor import extract_cross_references
from .document_graph import DocumentRecord, load_document
from .eda import KnowledgeGraphEDA
from .entity_canonicalizer import canonicalize, canonicalize_entities
from .graph_builder import KnowledgeGraphBuilder, build_graph_from_triples
from .knowledge_base import LegislativeKnowledgeBase
from .law_loader import LawRecord, load_one, load_real_laws
from .legislative_generator import RomanianLegislativeGenerator
from .llm_handler import (
    EmbeddingsHandler,
    LLMHandler,
    TripleExtractor,
    VectorStoreHandler,
)
from .ontology import LegislativeOntologyReasoner
from .pronoun_resolver import resolve_pronouns
from .similarity import SimilarityReport, compare_to_ontology, count_axiom_violations

__version__ = "2.0.0"
__author__ = "Maria"
__description__ = "Romanian Legislative Knowledge Graph Analysis System"

__all__ = [
    # Legislative Generation
    "RomanianLegislativeGenerator",
    # Graph Building
    "KnowledgeGraphBuilder",
    "build_graph_from_triples",
    # EDA
    "KnowledgeGraphEDA",
    # Ontology
    "LegislativeOntologyReasoner",
    # LLM
    "LLMHandler",
    "EmbeddingsHandler",
    "VectorStoreHandler",
    "TripleExtractor",
    # Knowledge Base
    "LegislativeKnowledgeBase",
    # Real Law Loader
    "load_real_laws",
    "load_one",
    "LawRecord",
    # Article Splitter
    "split_into_articles",
    # Cross-reference extractor
    "extract_cross_references",
    # Pronoun resolver
    "resolve_pronouns",
    # Entity canonicalizer
    "canonicalize",
    "canonicalize_entities",
    # Document subgraphs + similarity
    "DocumentRecord",
    "load_document",
    "SimilarityReport",
    "compare_to_ontology",
    "count_axiom_violations",
]
