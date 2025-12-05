# -*- coding: utf-8 -*-
"""
Module de détection d'objets.
Ce fichier maintient la rétrocompatibilité en réexportant depuis les nouveaux modules.
"""

# Réexporter depuis les nouveaux modules pour la rétrocompatibilité
from JM_C.src.core.constants import (
    COCO_LABELS,
    AVAILABLE_MODELS,
    get_label,
    get_available_models
)

from JM_C.src.utils.colors import (
    get_color_rgb,
    get_color_hex,
    get_color_rgba
)

from JM_C.src.core.data_types import (
    Detection,
    ModelInfo
)

from JM_C.src.core.detector import (
    ObjectDetector,
    get_model_info
)

from JM_C.src.utils.visualization import (
    draw_detections,
    draw_masks_only,
    create_mask_overlay
)

from JM_C.src.utils.image_utils import (
    load_image,
    image_to_array,
    array_to_image
)


__all__ = [
    # Constantes
    'COCO_LABELS',
    'AVAILABLE_MODELS',
    'get_label',
    'get_available_models',
    # Couleurs
    'get_color_rgb',
    'get_color_hex',
    'get_color_rgba',
    # Types
    'Detection',
    'ModelInfo',
    # Détecteur
    'ObjectDetector',
    'get_model_info',
    # Visualisation
    'draw_detections',
    'draw_masks_only',
    'create_mask_overlay',
    # Images
    'load_image',
    'image_to_array',
    'array_to_image',
]
