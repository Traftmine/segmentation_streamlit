# -*- coding: utf-8 -*-
"""
Configuration de l'application.
"""

from pathlib import Path

# =============================================================================
# CHEMINS
# =============================================================================

# Répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent

# Répertoire des données
DATA_DIR = ROOT_DIR / "data"

# Répertoire source
SRC_DIR = ROOT_DIR / "src"


# =============================================================================
# CONFIGURATION STREAMLIT
# =============================================================================

STREAMLIT_CONFIG = {
    "page_title": "Détection d'Objets",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}


# =============================================================================
# CONFIGURATION DES MODÈLES
# =============================================================================

# Seuil de confiance par défaut
DEFAULT_THRESHOLD = 0.5

# Taille minimale de la police pour les labels
MIN_FONT_SIZE = 12

# Taille maximale de la police pour les labels
MAX_FONT_SIZE = 24

# Taille par défaut de la police
DEFAULT_FONT_SIZE = 16

# Épaisseur des lignes des boîtes
BOX_LINE_WIDTH = 3


# =============================================================================
# FORMATS D'IMAGE SUPPORTÉS
# =============================================================================

SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
