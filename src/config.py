"""
Configuration settings for the Romanian Legislative Knowledge Graph project.
Centralizes all constants, paths, and model configurations.
"""

import os

# ==========================================
# Model Configuration
# ==========================================
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma2:9b"  # gemma2:9b, gemma2:27b, llama3.1:8b
LLM_TEMPERATURE = 0.0
LLM_TEMPERATURE_EXTRACTION = 0.1
CONFIDENCE_REGEX = 1.0
CONFIDENCE_LLM = 0.7

# ==========================================
# Paths Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = OUTPUT_DIR  # All data files go to output folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHROMA_DB_PATH = os.path.join(OUTPUT_DIR, "legislative_knowledge_db")
CHROMA_COLLECTION_NAME = "romanian_legislative_knowledge"

# ==========================================
# Legislative Data Configuration
# ==========================================
LEGISLATIVE_CORPUS_PATH = os.path.join(OUTPUT_DIR, "legislative_corpus.csv")
LEGISLATIVE_TRIPLES_PATH = os.path.join(OUTPUT_DIR, "legislative_triples.csv")

# ==========================================
# Visualization Configuration
# ==========================================
PLOT_STYLE = {
    "figure_size_large": (12, 6),
    "figure_size_medium": (10, 6),
    "figure_size_split": (12, 5),
    "dpi": 100,
    "color_primary": "skyblue",
    "color_secondary": "coral",
    "color_tertiary": "lightgreen",
    "edge_color": "black",
}


# Plot output filenames
def get_plot_path(name):
    """Generate plot output path."""
    return os.path.join(OUTPUT_DIR, f"{name}.png")


# ==========================================
# Legislative Ontology Configuration
# ==========================================

# Core legislative entities
LEGISLATIVE_ENTITIES = {
    "emitters": ["Parlamentul României", "Guvernul României", "Președintele României"],
    "ministries": [
        "Ministerul Cercetării, Inovării și Digitalizării",
        "Ministerul Educației",
        "Ministerul Sănătății",
        "Ministerul Finanțelor Publice",
        "Ministerul Justiției",
    ],
    "publishers": ["Monitorul Oficial al României"],
    "locations": ["București", "Palatul Parlamentului", "Palatul Cotroceni"],
}

# Functional properties (max cardinality = 1)
# A law can only be issued by ONE primary authority
LEGISLATIVE_FUNCTIONAL_RELATIONS = ["emis_de", "promulgat_de", "publicat_în"]

# Asymmetric properties (if A→B then B cannot→A)
# If Law A modifies Law B, then Law B cannot modify Law A
LEGISLATIVE_ASYMMETRIC_RELATIONS = ["modifică", "abroga", "completează"]

# Irreflexive properties (no self-loops)
# A law cannot modify itself, promulgate itself, etc.
LEGISLATIVE_IRREFLEXIVE_RELATIONS = ["modifică", "promulgat_de", "abroga", "completează", "emis_de"]

# Transitive properties
# If A modifies B and B modifies C, then A indirectly affects C
LEGISLATIVE_TRANSITIVE_RELATIONS = ["modifică"]

# Symmetric properties (if A→B then B→A)
# If Ministry A collaborates with Ministry B, then B collaborates with A
LEGISLATIVE_SYMMETRIC_RELATIONS = ["colaborează_cu"]

# Domain-specific constraints
LEGISLATIVE_CONSTRAINTS = {
    # Only these entities can emit laws
    "emis_de_domain": ["Parlamentul României", "Guvernul României"],
    # Only President can promulgate
    "promulgat_de_domain": ["Klaus Iohannis", "Președintele României"],
    # Only Official Monitor can publish
    "publicat_în_domain": ["Monitorul Oficial al României"],
}

# Legislative hierarchy (for temporal and authority reasoning)
LAW_HIERARCHY = {
    "Constituție": 1,  # Highest authority
    "Lege organică": 2,  # Organic laws
    "Lege": 3,  # Ordinary laws
    "Ordonanță de urgență": 4,  # Emergency ordinances
    "Ordonanță": 5,  # Government ordinances
    "Hotărâre de guvern": 6,  # Government decisions
}
