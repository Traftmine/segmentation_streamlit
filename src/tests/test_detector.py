# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module detector.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import ObjectDetector, get_model_info
from core.data_types import ModelInfo


class TestObjectDetectorInit:
    """Tests pour l'initialisation de ObjectDetector."""
    
    def test_init_with_valid_model(self):
        """Vérifie l'initialisation avec un modèle valide."""
        detector = ObjectDetector("SSD MobileNet V2")
        assert detector.model_name == "SSD MobileNet V2"
        assert detector.model is None  # Non chargé
    
    def test_init_with_invalid_model(self):
        """Vérifie qu'une erreur est levée pour un modèle invalide."""
        with pytest.raises(ValueError) as excinfo:
            ObjectDetector("Modèle Inexistant")
        assert "Modèle inconnu" in str(excinfo.value)
    
    def test_is_loaded_false_initially(self):
        """Vérifie que is_loaded retourne False initialement."""
        detector = ObjectDetector("SSD MobileNet V2")
        assert detector.is_loaded() is False
    
    def test_model_type_detection(self):
        """Vérifie le type de modèle pour la détection."""
        detector = ObjectDetector("SSD MobileNet V2")
        assert detector.model_type == "detection"
    
    def test_model_type_segmentation(self):
        """Vérifie le type de modèle pour la segmentation."""
        detector = ObjectDetector("Mask R-CNN Inception ResNet V2")
        assert detector.model_type == "segmentation"


class TestObjectDetectorPredict:
    """Tests pour la méthode predict."""
    
    @pytest.fixture
    def detector(self):
        """Crée un détecteur pour les tests."""
        return ObjectDetector("SSD MobileNet V2")
    
    def test_predict_without_loading_raises(self, detector):
        """Vérifie qu'une erreur est levée si le modèle n'est pas chargé."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError) as excinfo:
            detector.predict(image)
        assert "n'est pas chargé" in str(excinfo.value)


class TestGetModelInfo:
    """Tests pour la fonction get_model_info."""
    
    def test_returns_model_info(self):
        """Vérifie que la fonction retourne un ModelInfo."""
        info = get_model_info("SSD MobileNet V2")
        assert isinstance(info, ModelInfo)
    
    def test_model_info_fields(self):
        """Vérifie les champs du ModelInfo."""
        info = get_model_info("SSD MobileNet V2")
        assert info.name == "SSD MobileNet V2"
        assert "tfhub.dev" in info.url
        assert info.model_type in ["detection", "segmentation"]
        assert len(info.speed) > 0
        assert len(info.accuracy) > 0
        assert len(info.description) > 0
    
    def test_invalid_model_raises(self):
        """Vérifie qu'une erreur est levée pour un modèle invalide."""
        with pytest.raises(KeyError):
            get_model_info("Modèle Inexistant")
    
    def test_different_models_have_different_urls(self):
        """Vérifie que différents modèles ont différentes URLs."""
        info1 = get_model_info("SSD MobileNet V2")
        info2 = get_model_info("EfficientDet D0")
        assert info1.url != info2.url


class TestGenerateEllipseMask:
    """Tests pour la génération de masques elliptiques."""
    
    @pytest.fixture
    def detector(self):
        """Crée un détecteur pour les tests."""
        return ObjectDetector("SSD MobileNet V2")
    
    def test_mask_shape(self, detector):
        """Vérifie les dimensions du masque généré."""
        mask = detector._generate_ellipse_mask(10, 20, 50, 80, 100, 100)
        assert mask.shape == (100, 100)
    
    def test_mask_dtype(self, detector):
        """Vérifie le type de données du masque."""
        mask = detector._generate_ellipse_mask(0, 0, 50, 50, 100, 100)
        assert mask.dtype == np.float32
    
    def test_mask_values(self, detector):
        """Vérifie que le masque contient des valeurs 0.0 et 1.0."""
        mask = detector._generate_ellipse_mask(10, 10, 90, 90, 100, 100)
        unique_values = np.unique(mask)
        assert 0.0 in unique_values
        assert 1.0 in unique_values
    
    def test_mask_has_ellipse_inside_box(self, detector):
        """Vérifie que l'ellipse est à l'intérieur de la boîte."""
        left, top, right, bottom = 20, 30, 80, 70
        mask = detector._generate_ellipse_mask(left, top, right, bottom, 100, 100)
        
        # Les coins de la boîte doivent être vides
        assert mask[top, left] == 0.0
        assert mask[top, right-1] == 0.0
        assert mask[bottom-1, left] == 0.0
        assert mask[bottom-1, right-1] == 0.0
        
        # Le centre doit être rempli
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        assert mask[center_y, center_x] == 1.0
