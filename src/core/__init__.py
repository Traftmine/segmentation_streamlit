# -*- coding: utf-8 -*-
"""
Module core - Logique métier de la détection d'objets.

Contient:
- constants.py  : Labels COCO et modèles disponibles
- data_types.py : Types de données (Detection, ModelInfo)
- detector.py   : Classe ObjectDetector principale
"""

from .constants import (
    COCO_LABELS,
    AVAILABLE_MODELS,
)

from .data_types import (
    Detection,
    ModelInfo
)

from .detector import (
    ObjectDetector,
    get_model_info
)

__all__ = [
    'COCO_LABELS',
    'AVAILABLE_MODELS',
    'get_label',
    'get_available_models',
    'Detection',
    'ModelInfo',
    'ObjectDetector',
    'get_model_info',
]
