# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module helpers.
"""

import pytest
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_label, get_available_models


class TestGetLabel:
    """Tests pour la fonction get_label."""
    
    @pytest.fixture
    def sample_labels(self):
        """Fixture avec des labels de test."""
        return {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            17: 'cat',
            18: 'dog'
        }
    
    def test_returns_correct_label(self, sample_labels):
        """Vérifie que le bon label est retourné."""
        assert get_label(1, sample_labels) == 'person'
        assert get_label(17, sample_labels) == 'cat'
        assert get_label(18, sample_labels) == 'dog'
    
    def test_returns_inconnu_for_missing_id(self, sample_labels):
        """Vérifie que 'inconnu' est retourné pour un ID manquant."""
        assert get_label(999, sample_labels) == 'inconnu'
        assert get_label(0, sample_labels) == 'inconnu'
        assert get_label(-1, sample_labels) == 'inconnu'
    
    def test_empty_labels(self):
        """Vérifie le comportement avec un dictionnaire vide."""
        assert get_label(1, {}) == 'inconnu'
    
    def test_with_real_coco_labels(self):
        """Test avec les vrais labels COCO."""
        from core.constants import COCO_LABELS
        
        assert get_label(1, COCO_LABELS) == 'person'
        assert get_label(17, COCO_LABELS) == 'cat'
        assert get_label(18, COCO_LABELS) == 'dog'
        assert get_label(90, COCO_LABELS) == 'toothbrush'


class TestGetAvailableModels:
    """Tests pour la fonction get_available_models."""
    
    @pytest.fixture
    def sample_models(self):
        """Fixture avec des modèles de test."""
        return {
            "Model A": {"url": "http://a.com", "type": "detection"},
            "Model B": {"url": "http://b.com", "type": "segmentation"}
        }
    
    def test_returns_same_dict(self, sample_models):
        """Vérifie que la fonction retourne le même dictionnaire."""
        result = get_available_models(sample_models)
        assert result == sample_models
    
    def test_returns_empty_for_empty_input(self):
        """Vérifie le comportement avec un dictionnaire vide."""
        assert get_available_models({}) == {}
    
    def test_with_real_models(self):
        """Test avec les vrais modèles disponibles."""
        from core.constants import AVAILABLE_MODELS
        
        result = get_available_models(AVAILABLE_MODELS)
        assert "SSD MobileNet V2" in result
        assert "url" in result["SSD MobileNet V2"]
