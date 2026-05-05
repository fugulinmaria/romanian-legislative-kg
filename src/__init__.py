"""
Romanian Legislative Knowledge Graph Analysis Package
Provides modular tools for generating, extracting, and reasoning
over Romanian legislative knowledge.
"""

from .article_splitter import split_into_articles
from .config import *  # noqa: F403
from .cross_reference_extractor import extract_cross_references
from .data_loader import TripleDataLoader, load_legislative_triples, save_legislative_triples
from .eda import KnowledgeGraphEDA
from .graph_builder import KnowledgeGraphBuilder, build_graph_from_triples
from .knowledge_base import LegislativeKnowledgeBase, create_knowledge_base
from .law_loader import LawRecord, load_one, load_real_laws
from .legislative_generator import RomanianLegislativeGenerator, generate_romanian_law
from .llm_handler import (
    EmbeddingsHandler,
    KnowledgeGraphEvaluator,
    LLMHandler,
    TripleExtractor,
    VectorStoreHandler,
    init_llm_models,
)
from .ontology import LegislativeOntologyReasoner

__version__ = "2.0.0"
__author__ = "Maria"
__description__ = "Romanian Legislative Knowledge Graph Analysis System"

__all__ = [
    # Data Loading
    "TripleDataLoader",
    "load_legislative_triples",
    "save_legislative_triples",
    # Legislative Generation
    "RomanianLegislativeGenerator",
    "generate_romanian_law",
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
    "KnowledgeGraphEvaluator",
    "init_llm_models",
    # Knowledge Base
    "LegislativeKnowledgeBase",
    "create_knowledge_base",
    # Real Law Loader
    "load_real_laws",
    "load_one",
    "LawRecord",
    # Article Splitter
    "split_into_articles",
    # Cross-reference extractor
    "extract_cross_references",
]
