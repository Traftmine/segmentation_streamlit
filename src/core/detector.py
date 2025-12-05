# -*- coding: utf-8 -*-
"""
Classe principale de détection d'objets.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .constants import AVAILABLE_MODELS, COCO_LABELS
from .data_types import Detection, ModelInfo
from utils.helpers import get_label


class ObjectDetector:
    """
    Classe pour la détection d'objets utilisant TensorFlow Hub.
    Supporte la détection et la segmentation d'instance.
    """
    
    def __init__(self, model_name: str):
        """
        Initialise le détecteur avec un modèle.
        
        Args:
            model_name: Nom du modèle (clé de AVAILABLE_MODELS)
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        model_info = AVAILABLE_MODELS[model_name]
        self.model_name = model_name
        self.model_url = model_info["url"]
        self.model_type = model_info["type"]
        self.model = None
    
    def load(self) -> None:
        """Charge le modèle depuis TensorFlow Hub."""
        self.model = hub.load(self.model_url)
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self.model is not None
    
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Exécute la prédiction sur une image.
        
        Args:
            image: Image sous forme de tableau numpy (H, W, 3)
            
        Returns:
            Dictionnaire contenant les résultats de détection
        """
        if not self.is_loaded():
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load() d'abord.")
        
        # Convertir en uint8 si nécessaire
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        results = self.model(input_tensor)
        
        return {key: value.numpy() for key, value in results.items()}
    
    def detect(
        self, 
        image: np.ndarray, 
        threshold: float = 0.5,
        max_detections: int = 100,
        generate_approx_masks: bool = True
    ) -> List[Detection]:
        """
        Détecte les objets dans une image.
        
        Args:
            image: Image sous forme de tableau numpy (H, W, 3)
            threshold: Seuil de confiance minimum (0.0 à 1.0)
            max_detections: Nombre maximum de détections
            generate_approx_masks: Génère des masques approximatifs si le modèle n'en fournit pas
            
        Returns:
            Liste des détections
        """
        results = self.predict(image)
        
        if 'detection_boxes' not in results:
            raise ValueError("Format de sortie du modèle non reconnu")
        
        boxes = results['detection_boxes'][0]
        classes = results['detection_classes'][0].astype(int)
        scores = results['detection_scores'][0]
        
        # Vérifier si des masques sont disponibles (modèles Mask R-CNN)
        masks = None
        has_native_masks = 'detection_masks' in results
        if has_native_masks:
            masks = results['detection_masks'][0]
        
        height, width = image.shape[:2]
        detections = []
        
        for i in range(min(len(scores), max_detections)):
            if scores[i] >= threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Coordonnées en pixels
                left = int(xmin * width)
                top = int(ymin * height)
                right = int(xmax * width)
                bottom = int(ymax * height)
                
                # Créer le masque
                mask = None
                if has_native_masks and masks is not None:
                    # Masque natif du modèle (Mask R-CNN)
                    mask = self._process_mask(
                        masks[i], 
                        boxes[i], 
                        height, 
                        width
                    )
                elif generate_approx_masks:
                    # Générer un masque approximatif (ellipse dans la boîte)
                    mask = self._generate_ellipse_mask(
                        left, top, right, bottom,
                        height, width
                    )
                
                detection = Detection(
                    class_id=int(classes[i]),
                    class_name=get_label(int(classes[i]), COCO_LABELS),
                    confidence=float(scores[i]),
                    box=(left, top, right, bottom),
                    mask=mask
                )
                detections.append(detection)
        
        return detections
    
    def _generate_ellipse_mask(
        self,
        left: int, top: int, right: int, bottom: int,
        image_height: int, image_width: int
    ) -> np.ndarray:
        """
        Génère un masque elliptique approximatif basé sur la boîte englobante.
        
        Args:
            left, top, right, bottom: Coordonnées de la boîte en pixels
            image_height: Hauteur de l'image
            image_width: Largeur de l'image
            
        Returns:
            Masque binaire (float32, 0.0 ou 1.0)
        """
        # Créer une image pour dessiner l'ellipse
        mask_img = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Dessiner une ellipse remplie dans la boîte
        # Réduire légèrement pour un effet plus naturel
        padding_x = int((right - left) * 0.05)
        padding_y = int((bottom - top) * 0.05)
        draw.ellipse(
            [left + padding_x, top + padding_y, 
             right - padding_x, bottom - padding_y],
            fill=255
        )
        
        return np.array(mask_img, dtype=np.float32) / 255.0
    
    def _process_mask(
        self, 
        mask: np.ndarray, 
        box: np.ndarray, 
        image_height: int, 
        image_width: int
    ) -> np.ndarray:
        """
        Traite et redimensionne un masque à la taille de l'image.
        
        Args:
            mask: Masque brut du modèle
            box: Boîte englobante normalisée [ymin, xmin, ymax, xmax]
            image_height: Hauteur de l'image
            image_width: Largeur de l'image
            
        Returns:
            Masque binaire de la taille de l'image
        """
        ymin, xmin, ymax, xmax = box
        
        # Convertir en coordonnées pixels
        y1 = int(ymin * image_height)
        x1 = int(xmin * image_width)
        y2 = int(ymax * image_height)
        x2 = int(xmax * image_width)
        
        # Dimensions de la boîte
        box_height = max(y2 - y1, 1)
        box_width = max(x2 - x1, 1)
        
        # Redimensionner le masque à la taille de la boîte
        mask_resized = tf.image.resize(
            mask[..., tf.newaxis], 
            [box_height, box_width],
            method='bilinear'
        ).numpy()[:, :, 0]
        
        # Créer un masque de la taille de l'image
        full_mask = np.zeros((image_height, image_width), dtype=np.float32)
        
        # Placer le masque redimensionné dans l'image
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image_height, y2)
        x2 = min(image_width, x2)
        
        mask_h = y2 - y1
        mask_w = x2 - x1
        
        if mask_h > 0 and mask_w > 0:
            full_mask[y1:y2, x1:x2] = mask_resized[:mask_h, :mask_w]
        
        return full_mask


def get_model_info(model_name: str) -> ModelInfo:
    """Retourne les informations sur un modèle."""
    info = AVAILABLE_MODELS[model_name]
    return ModelInfo(
        name=model_name,
        url=info["url"],
        model_type=info["type"],
        speed=info["speed"],
        accuracy=info["accuracy"],
        description=info["description"]
    )
