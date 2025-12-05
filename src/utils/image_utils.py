# -*- coding: utf-8 -*-
"""
Utilitaires pour la manipulation d'images.
"""

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    """
    Charge une image depuis un fichier.
    
    Args:
        path: Chemin vers le fichier image
        
    Returns:
        Image PIL en mode RGB
    """
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def image_to_array(image: Image.Image) -> np.ndarray:
    """
    Convertit une image PIL en tableau numpy.
    
    Args:
        image: Image PIL
        
    Returns:
        Tableau numpy (H, W, 3)
    """
    return np.array(image)


def array_to_image(array: np.ndarray) -> Image.Image:
    """
    Convertit un tableau numpy en image PIL.
    
    Args:
        array: Tableau numpy (H, W, 3)
        
    Returns:
        Image PIL
    """
    return Image.fromarray(array)
