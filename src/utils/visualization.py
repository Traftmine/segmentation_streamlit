# -*- coding: utf-8 -*-
"""
Fonctions de visualisation des détections.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import Detection
from .colors import get_color_hex, get_color_rgba


# =============================================================================
# FONCTIONS PRIVÉES
# =============================================================================

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Charge une police système."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    
    return ImageFont.load_default()


def _draw_mask(
    layer: Image.Image, 
    detection: Detection, 
    alpha: int
) -> None:
    """Dessine un masque de segmentation sur un calque."""
    if detection.mask is None:
        return
    
    color = get_color_rgba(detection.class_id, alpha)
    mask = detection.mask
    
    # Créer une image colorée pour le masque
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_binary, mode='L')
    
    # Créer une image de la couleur du masque
    color_image = Image.new('RGBA', layer.size, color)
    
    # Appliquer le masque
    layer.paste(color_image, (0, 0), mask_image)


def _draw_box(
    draw: ImageDraw.Draw, 
    detection: Detection, 
    color: str, 
    thickness: int
) -> None:
    """Dessine une boîte englobante."""
    left, top, right, bottom = detection.box
    draw.rectangle([left, top, right, bottom], outline=color, width=thickness)


def _draw_label(
    draw: ImageDraw.Draw, 
    detection: Detection, 
    color: str, 
    font: ImageFont.FreeTypeFont
) -> None:
    """Dessine un label avec fond."""
    left, top, right, bottom = detection.box
    text = f"{detection.class_name}: {detection.confidence:.0%}"
    
    # Calculer la position du texte
    bbox = draw.textbbox((left, top - 25), text, font=font)
    
    # Ajuster si le texte sort de l'image
    if bbox[1] < 0:
        text_y = bottom + 5
        bbox = draw.textbbox((left, text_y), text, font=font)
    else:
        text_y = top - 25
    
    # Dessiner le fond
    padding = 2
    draw.rectangle(
        [bbox[0] - padding, bbox[1] - padding, 
         bbox[2] + padding, bbox[3] + padding], 
        fill=color
    )
    
    # Dessiner le texte
    draw.text((left, text_y), text, fill='white', font=font)


# =============================================================================
# FONCTIONS PUBLIQUES
# =============================================================================

def draw_detections(
    image: Image.Image, 
    detections: List[Detection],
    show_boxes: bool = True,
    show_labels: bool = True,
    show_masks: bool = True,
    mask_alpha: int = 100,
    box_thickness: int = 3,
    font_size: int = 16
) -> Image.Image:
    """
    Dessine les détections sur une image.
    
    Args:
        image: Image PIL
        detections: Liste des détections
        show_boxes: Afficher les boîtes englobantes
        show_labels: Afficher les labels
        show_masks: Afficher les masques de segmentation
        mask_alpha: Opacité des masques (0-255)
        box_thickness: Épaisseur des boîtes
        font_size: Taille de la police
        
    Returns:
        Image avec les détections dessinées
    """
    result = image.copy().convert('RGBA')
    
    # Créer un calque pour les masques
    if show_masks:
        mask_layer = Image.new('RGBA', result.size, (0, 0, 0, 0))
        
        for detection in detections:
            if detection.mask is not None:
                _draw_mask(mask_layer, detection, mask_alpha)
        
        # Fusionner le calque des masques
        result = Image.alpha_composite(result, mask_layer)
    
    # Dessiner les boîtes et labels
    result_rgb = result.convert('RGB')
    draw = ImageDraw.Draw(result_rgb)
    font = _load_font(font_size)
    
    for detection in detections:
        color = get_color_hex(detection.class_id)
        
        if show_boxes:
            _draw_box(draw, detection, color, box_thickness)
        
        if show_labels:
            _draw_label(draw, detection, color, font)
    
    return result_rgb


def draw_masks_only(
    image: Image.Image,
    detections: List[Detection],
    alpha: int = 150
) -> Image.Image:
    """
    Dessine uniquement les masques de segmentation.
    
    Args:
        image: Image PIL
        detections: Liste des détections
        alpha: Opacité des masques
        
    Returns:
        Image avec les masques
    """
    result = image.copy().convert('RGBA')
    mask_layer = Image.new('RGBA', result.size, (0, 0, 0, 0))
    
    for detection in detections:
        if detection.mask is not None:
            _draw_mask(mask_layer, detection, alpha)
    
    result = Image.alpha_composite(result, mask_layer)
    return result.convert('RGB')


def create_mask_overlay(
    image: Image.Image,
    detections: List[Detection],
    selected_classes: Optional[List[int]] = None
) -> Image.Image:
    """
    Crée un overlay de masques pour des classes spécifiques.
    
    Args:
        image: Image PIL
        detections: Liste des détections
        selected_classes: IDs des classes à afficher (None = toutes)
        
    Returns:
        Image avec overlay
    """
    result = image.copy().convert('RGBA')
    
    for detection in detections:
        if detection.mask is None:
            continue
        
        if selected_classes is not None and detection.class_id not in selected_classes:
            continue
        
        mask = detection.mask
        color = get_color_rgba(detection.class_id, 128)
        
        # Créer le masque binaire
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_binary, mode='L')
        
        # Créer l'overlay coloré
        overlay = Image.new('RGBA', result.size, color)
        result = Image.composite(overlay, result, mask_image)
    
    return result.convert('RGB')
