# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module image_utils.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import load_image, image_to_array, array_to_image


class TestLoadImage:
    """Tests pour la fonction load_image."""
    
    @pytest.fixture
    def temp_rgb_image(self):
        """Crée une image RGB temporaire."""
        img = Image.new('RGB', (100, 100), color='red')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_rgba_image(self):
        """Crée une image RGBA temporaire."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_grayscale_image(self):
        """Crée une image en niveaux de gris temporaire."""
        img = Image.new('L', (100, 100), color=128)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            yield f.name
        os.unlink(f.name)
    
    def test_load_rgb_image(self, temp_rgb_image):
        """Vérifie le chargement d'une image RGB."""
        img = load_image(temp_rgb_image)
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'
        assert img.size == (100, 100)
    
    def test_load_rgba_converts_to_rgb(self, temp_rgba_image):
        """Vérifie que RGBA est converti en RGB."""
        img = load_image(temp_rgba_image)
        assert img.mode == 'RGB'
    
    def test_load_grayscale_converts_to_rgb(self, temp_grayscale_image):
        """Vérifie que les niveaux de gris sont convertis en RGB."""
        img = load_image(temp_grayscale_image)
        assert img.mode == 'RGB'
    
    def test_load_nonexistent_file_raises(self):
        """Vérifie qu'une erreur est levée pour un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_image('/path/to/nonexistent/image.png')


class TestImageToArray:
    """Tests pour la fonction image_to_array."""
    
    @pytest.fixture
    def sample_image(self):
        """Crée une image PIL de test."""
        return Image.new('RGB', (50, 100), color='blue')
    
    def test_returns_numpy_array(self, sample_image):
        """Vérifie que la fonction retourne un tableau numpy."""
        result = image_to_array(sample_image)
        assert isinstance(result, np.ndarray)
    
    def test_correct_shape(self, sample_image):
        """Vérifie les dimensions du tableau."""
        result = image_to_array(sample_image)
        # (height, width, channels)
        assert result.shape == (100, 50, 3)
    
    def test_correct_dtype(self, sample_image):
        """Vérifie le type de données."""
        result = image_to_array(sample_image)
        assert result.dtype == np.uint8
    
    def test_pixel_values(self):
        """Vérifie les valeurs des pixels."""
        img = Image.new('RGB', (10, 10), color=(255, 128, 0))
        arr = image_to_array(img)
        
        # Tous les pixels doivent être (255, 128, 0)
        assert np.all(arr[:, :, 0] == 255)  # Rouge
        assert np.all(arr[:, :, 1] == 128)  # Vert
        assert np.all(arr[:, :, 2] == 0)    # Bleu


class TestArrayToImage:
    """Tests pour la fonction array_to_image."""
    
    @pytest.fixture
    def sample_array(self):
        """Crée un tableau numpy de test."""
        arr = np.zeros((100, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Rouge
        return arr
    
    def test_returns_pil_image(self, sample_array):
        """Vérifie que la fonction retourne une image PIL."""
        result = array_to_image(sample_array)
        assert isinstance(result, Image.Image)
    
    def test_correct_size(self, sample_array):
        """Vérifie les dimensions de l'image."""
        result = array_to_image(sample_array)
        # PIL size is (width, height)
        assert result.size == (50, 100)
    
    def test_correct_mode(self, sample_array):
        """Vérifie le mode de l'image."""
        result = array_to_image(sample_array)
        assert result.mode == 'RGB'
    
    def test_roundtrip_conversion(self):
        """Vérifie la conversion aller-retour."""
        original = Image.new('RGB', (30, 40), color=(100, 150, 200))
        array = image_to_array(original)
        restored = array_to_image(array)
        
        # Vérifier que les pixels sont identiques
        assert list(original.getdata()) == list(restored.getdata())
