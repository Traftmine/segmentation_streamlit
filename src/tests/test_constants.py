# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module constants.
"""

import pytest
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import COCO_LABELS, AVAILABLE_MODELS


class TestCocoLabels:
    """Tests pour les labels COCO."""
    
    def test_is_dict(self):
        """Vérifie que COCO_LABELS est un dictionnaire."""
        assert isinstance(COCO_LABELS, dict)
    
    def test_has_80_classes(self):
        """Vérifie qu'il y a 80 classes COCO."""
        assert len(COCO_LABELS) == 80
    
    def test_keys_are_integers(self):
        """Vérifie que les clés sont des entiers."""
        for key in COCO_LABELS.keys():
            assert isinstance(key, int)
    
    def test_values_are_strings(self):
        """Vérifie que les valeurs sont des chaînes."""
        for value in COCO_LABELS.values():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_key_range(self):
        """Vérifie que les IDs sont dans la plage attendue."""
        for key in COCO_LABELS.keys():
            assert 1 <= key <= 90
    
    def test_common_classes_exist(self):
        """Vérifie que les classes communes existent."""
        assert COCO_LABELS[1] == 'person'
        assert COCO_LABELS[17] == 'cat'
        assert COCO_LABELS[18] == 'dog'
        assert COCO_LABELS[3] == 'car'
    
    def test_no_duplicate_labels(self):
        """Vérifie qu'il n'y a pas de labels dupliqués."""
        labels = list(COCO_LABELS.values())
        assert len(labels) == len(set(labels))


class TestAvailableModels:
    """Tests pour les modèles disponibles."""
    
    def test_is_dict(self):
        """Vérifie que AVAILABLE_MODELS est un dictionnaire."""
        assert isinstance(AVAILABLE_MODELS, dict)
    
    def test_has_models(self):
        """Vérifie qu'il y a des modèles définis."""
        assert len(AVAILABLE_MODELS) > 0
    
    def test_model_names_are_strings(self):
        """Vérifie que les noms de modèles sont des chaînes."""
        for name in AVAILABLE_MODELS.keys():
            assert isinstance(name, str)
            assert len(name) > 0
    
    def test_each_model_has_required_fields(self):
        """Vérifie que chaque modèle a les champs requis."""
        required_fields = ['url', 'type', 'speed', 'accuracy', 'description']
        
        for model_name, model_info in AVAILABLE_MODELS.items():
            assert isinstance(model_info, dict), f"{model_name} n'est pas un dict"
            for field in required_fields:
                assert field in model_info, f"{model_name} manque le champ '{field}'"
    
    def test_model_urls_are_valid(self):
        """Vérifie que les URLs de modèles sont valides."""
        for model_name, model_info in AVAILABLE_MODELS.items():
            url = model_info['url']
            assert url.startswith('https://tfhub.dev/'), \
                f"{model_name} a une URL invalide: {url}"
    
    def test_model_types_are_valid(self):
        """Vérifie que les types de modèles sont valides."""
        valid_types = {'detection', 'segmentation'}
        
        for model_name, model_info in AVAILABLE_MODELS.items():
            assert model_info['type'] in valid_types, \
                f"{model_name} a un type invalide: {model_info['type']}"
    
    def test_ssd_models_exist(self):
        """Vérifie que les modèles SSD existent."""
        ssd_models = [m for m in AVAILABLE_MODELS.keys() if 'SSD' in m]
        assert len(ssd_models) > 0
    
    def test_efficientdet_models_exist(self):
        """Vérifie que les modèles EfficientDet existent."""
        efficientdet_models = [m for m in AVAILABLE_MODELS.keys() if 'EfficientDet' in m]
        assert len(efficientdet_models) > 0
    
    def test_mask_rcnn_supports_masks(self):
        """Vérifie que Mask R-CNN supporte les masques."""
        mask_models = [m for m in AVAILABLE_MODELS.keys() if 'Mask' in m]
        for model_name in mask_models:
            model_info = AVAILABLE_MODELS[model_name]
            assert model_info.get('supports_masks', False) is True, \
                f"{model_name} devrait supporter les masques"
            assert model_info['type'] == 'segmentation'
