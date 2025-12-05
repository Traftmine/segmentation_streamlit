# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module data_types.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import Detection, ModelInfo


class TestDetection:
    """Tests pour la classe Detection."""
    
    @pytest.fixture
    def sample_detection(self):
        """Fixture avec une détection de test."""
        return Detection(
            class_id=1,
            class_name='person',
            confidence=0.95,
            box=(10, 20, 100, 200)
        )
    
    @pytest.fixture
    def detection_with_mask(self):
        """Fixture avec une détection incluant un masque."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        return Detection(
            class_id=17,
            class_name='cat',
            confidence=0.87,
            box=(50, 50, 150, 150),
            mask=mask
        )
    
    def test_creation_without_mask(self, sample_detection):
        """Vérifie la création d'une détection sans masque."""
        assert sample_detection.class_id == 1
        assert sample_detection.class_name == 'person'
        assert sample_detection.confidence == 0.95
        assert sample_detection.box == (10, 20, 100, 200)
        assert sample_detection.mask is None
    
    def test_creation_with_mask(self, detection_with_mask):
        """Vérifie la création d'une détection avec masque."""
        assert detection_with_mask.class_id == 17
        assert detection_with_mask.mask is not None
        assert detection_with_mask.mask.shape == (100, 100)
    
    def test_to_dict_without_mask(self, sample_detection):
        """Vérifie la conversion en dictionnaire sans masque."""
        result = sample_detection.to_dict()
        
        assert isinstance(result, dict)
        assert result['class_id'] == 1
        assert result['class'] == 'person'
        assert result['confidence'] == 0.95
        assert result['box'] == [10, 20, 100, 200]
        assert result['has_mask'] is False
    
    def test_to_dict_with_mask(self, detection_with_mask):
        """Vérifie la conversion en dictionnaire avec masque."""
        result = detection_with_mask.to_dict()
        
        assert result['has_mask'] is True
        assert result['class'] == 'cat'
    
    def test_box_is_list_in_dict(self, sample_detection):
        """Vérifie que la box est convertie en liste dans le dict."""
        result = sample_detection.to_dict()
        assert isinstance(result['box'], list)
    
    def test_confidence_range(self):
        """Vérifie que la confiance peut être dans la plage [0, 1]."""
        det_low = Detection(1, 'test', 0.0, (0, 0, 10, 10))
        det_high = Detection(1, 'test', 1.0, (0, 0, 10, 10))
        
        assert det_low.confidence == 0.0
        assert det_high.confidence == 1.0


class TestModelInfo:
    """Tests pour la classe ModelInfo."""
    
    @pytest.fixture
    def sample_model_info(self):
        """Fixture avec des infos modèle de test."""
        return ModelInfo(
            name="Test Model",
            url="https://example.com/model",
            model_type="detection",
            speed="⚡ Très rapide",
            accuracy="★★★☆☆",
            description="Un modèle de test"
        )
    
    def test_creation(self, sample_model_info):
        """Vérifie la création d'un ModelInfo."""
        assert sample_model_info.name == "Test Model"
        assert sample_model_info.url == "https://example.com/model"
        assert sample_model_info.model_type == "detection"
        assert sample_model_info.speed == "⚡ Très rapide"
        assert sample_model_info.accuracy == "★★★☆☆"
        assert sample_model_info.description == "Un modèle de test"
    
    def test_model_types(self):
        """Vérifie différents types de modèles."""
        detection_model = ModelInfo(
            "M1", "url1", "detection", "fast", "★★★", "desc"
        )
        segmentation_model = ModelInfo(
            "M2", "url2", "segmentation", "slow", "★★★★★", "desc"
        )
        
        assert detection_model.model_type == "detection"
        assert segmentation_model.model_type == "segmentation"
