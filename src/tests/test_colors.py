# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module colors.
"""

import pytest
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.colors import (
    generate_colors, 
    get_color_rgb, 
    get_color_hex, 
    get_color_rgba,
    CLASS_COLORS
)


class TestGenerateColors:
    """Tests pour la fonction generate_colors."""
    
    def test_generate_correct_number_of_colors(self):
        """Vérifie que le bon nombre de couleurs est généré."""
        assert len(generate_colors(10)) == 10
        assert len(generate_colors(1)) == 1
        assert len(generate_colors(100)) == 100
    
    def test_generate_zero_colors(self):
        """Vérifie le comportement avec 0 couleurs."""
        assert len(generate_colors(0)) == 0
    
    def test_colors_are_rgb_tuples(self):
        """Vérifie que les couleurs sont des tuples RGB valides."""
        colors = generate_colors(5)
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            r, g, b = color
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
    
    def test_colors_are_distinct(self):
        """Vérifie que les couleurs générées sont distinctes."""
        colors = generate_colors(10)
        # Convertir en set pour vérifier l'unicité
        unique_colors = set(colors)
        assert len(unique_colors) == len(colors)


class TestGetColorRgb:
    """Tests pour la fonction get_color_rgb."""
    
    def test_returns_tuple(self):
        """Vérifie que la fonction retourne un tuple."""
        result = get_color_rgb(1)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_values_in_valid_range(self):
        """Vérifie que les valeurs RGB sont dans la plage valide."""
        for class_id in range(20):
            r, g, b = get_color_rgb(class_id)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
    
    def test_wraps_around_for_large_ids(self):
        """Vérifie que les IDs larges utilisent le modulo."""
        # CLASS_COLORS a 100 couleurs
        color_0 = get_color_rgb(0)
        color_100 = get_color_rgb(100)
        assert color_0 == color_100
    
    def test_different_ids_different_colors(self):
        """Vérifie que différents IDs donnent différentes couleurs."""
        assert get_color_rgb(1) != get_color_rgb(2)
        assert get_color_rgb(0) != get_color_rgb(50)


class TestGetColorHex:
    """Tests pour la fonction get_color_hex."""
    
    def test_returns_hex_string(self):
        """Vérifie que la fonction retourne une chaîne hexadécimale."""
        result = get_color_hex(1)
        assert isinstance(result, str)
        assert result.startswith('#')
        assert len(result) == 7  # #RRGGBB
    
    def test_hex_format_valid(self):
        """Vérifie que le format hexadécimal est valide."""
        for class_id in range(10):
            hex_color = get_color_hex(class_id)
            # Doit pouvoir être converti en entier
            int(hex_color[1:], 16)
    
    def test_consistent_with_rgb(self):
        """Vérifie la cohérence avec get_color_rgb."""
        for class_id in range(5):
            r, g, b = get_color_rgb(class_id)
            expected_hex = f'#{r:02x}{g:02x}{b:02x}'
            assert get_color_hex(class_id) == expected_hex


class TestGetColorRgba:
    """Tests pour la fonction get_color_rgba."""
    
    def test_returns_tuple_with_alpha(self):
        """Vérifie que la fonction retourne un tuple RGBA."""
        result = get_color_rgba(1)
        assert isinstance(result, tuple)
        assert len(result) == 4
    
    def test_default_alpha(self):
        """Vérifie la valeur alpha par défaut."""
        r, g, b, a = get_color_rgba(1)
        assert a == 100
    
    def test_custom_alpha(self):
        """Vérifie l'alpha personnalisé."""
        r, g, b, a = get_color_rgba(1, alpha=200)
        assert a == 200
    
    def test_rgb_consistent_with_get_color_rgb(self):
        """Vérifie la cohérence des valeurs RGB."""
        for class_id in range(5):
            rgb = get_color_rgb(class_id)
            rgba = get_color_rgba(class_id)
            assert rgb == rgba[:3]


class TestClassColors:
    """Tests pour la constante CLASS_COLORS."""
    
    def test_has_100_colors(self):
        """Vérifie qu'il y a 100 couleurs pré-générées."""
        assert len(CLASS_COLORS) == 100
    
    def test_all_valid_rgb(self):
        """Vérifie que toutes les couleurs sont des RGB valides."""
        for color in CLASS_COLORS:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for value in color:
                assert 0 <= value <= 255
