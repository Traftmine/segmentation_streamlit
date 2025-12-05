# -*- coding: utf-8 -*-
"""
Fonctions utilitaires génériques.
"""

from typing import Dict


def get_label(class_id: int, labels: Dict[int, str]) -> str:
    """
    Retourne le label pour un ID de classe donné.
    
    Args:
        class_id: L'ID de la classe (1-90 pour COCO)
        labels: Dictionnaire de labels {id: nom}
        
    Returns:
        Le nom de la classe ou 'inconnu' si non trouvé
    """
    return labels.get(class_id, 'inconnu')


def get_available_models(models: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Retourne les modèles disponibles.
    
    Args:
        models: Dictionnaire des modèles disponibles
        
    Returns:
        Le dictionnaire des modèles (peut être filtré si nécessaire)
    """
    return models
