# -*- coding: utf-8 -*-
"""
Configuration et fixtures partagées pour pytest.
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Ajouter le dossier src au path pour tous les tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_rgb_image():
    """Crée une image RGB de test."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_numpy_image():
    """Crée un tableau numpy représentant une image."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :, 0] = 255  # Moitié supérieure rouge
    arr[50:, :, 2] = 255  # Moitié inférieure bleue
    return arr


@pytest.fixture
def sample_detection():
    """Crée une détection de test."""
    from core.data_types import Detection
    return Detection(
        class_id=1,
        class_name='person',
        confidence=0.95,
        box=(10, 20, 100, 200)
    )


@pytest.fixture
def sample_detections():
    """Crée une liste de détections de test."""
    from core.data_types import Detection
    return [
        Detection(1, 'person', 0.95, (10, 20, 100, 200)),
        Detection(17, 'cat', 0.87, (150, 50, 250, 180)),
        Detection(18, 'dog', 0.75, (300, 100, 400, 300)),
    ]


@pytest.fixture
def coco_labels():
    """Retourne les labels COCO."""
    from core.constants import COCO_LABELS
    return COCO_LABELS


@pytest.fixture
def available_models():
    """Retourne les modèles disponibles."""
    from core.constants import AVAILABLE_MODELS
    return AVAILABLE_MODELS
