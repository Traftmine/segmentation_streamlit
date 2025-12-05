# -*- coding: utf-8 -*-
"""
Types de données pour la détection d'objets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class Detection:
    """Représente une détection d'objet."""
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]  # (left, top, right, bottom)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict:
        """Convertit la détection en dictionnaire."""
        return {
            'class_id': self.class_id,
            'class': self.class_name,
            'confidence': self.confidence,
            'box': list(self.box),
            'has_mask': self.mask is not None
        }


@dataclass
class ModelInfo:
    """Informations sur un modèle."""
    name: str
    url: str
    model_type: str
    speed: str
    accuracy: str
    description: str
