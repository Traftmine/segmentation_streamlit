# -*- coding: utf-8 -*-
"""
Gestion des couleurs pour la visualisation.
"""

import colorsys
from typing import List, Tuple


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Génère n couleurs distinctes en RGB.
    
    Args:
        n: Nombre de couleurs à générer
        
    Returns:
        Liste de tuples RGB
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


# Palette de 100 couleurs pré-générées pour les classes
CLASS_COLORS = generate_colors(100)


def get_color_rgb(class_id: int) -> Tuple[int, int, int]:
    """Retourne une couleur RGB pour un ID de classe."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def get_color_hex(class_id: int) -> str:
    """Retourne une couleur hexadécimale pour un ID de classe."""
    r, g, b = get_color_rgb(class_id)
    return f'#{r:02x}{g:02x}{b:02x}'


def get_color_rgba(class_id: int, alpha: int = 100) -> Tuple[int, int, int, int]:
    """Retourne une couleur RGBA pour un ID de classe."""
    r, g, b = get_color_rgb(class_id)
    return (r, g, b, alpha)
