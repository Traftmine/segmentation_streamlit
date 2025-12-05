# -*- coding: utf-8 -*-
"""
Application Streamlit pour la détection d'objets.
Interface utilisateur avancée avec support des masques.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from core.detector import ObjectDetector
from utils.image_utils import image_to_array
from ui.styles import inject_css
from ui.ui_components import (
    render_sidebar,
    render_header,
    render_image_upload,
    render_detection_results,
    render_comparison_view,
    render_footer
)


# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🔍 Détection d'Objets IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecter les styles CSS
inject_css()


# =============================================================================
# CACHE
# =============================================================================

@st.cache_resource
def load_detector(model_name: str) -> ObjectDetector:
    """Charge et met en cache le détecteur."""
    detector = ObjectDetector(model_name)
    detector.load()
    return detector


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Point d'entrée principal."""
    
    # Sidebar - Configuration
    config = render_sidebar()
    
    # Header
    render_header()
    
    # Charger le modèle
    try:
        with st.spinner(f"Chargement du modèle {config['model_name']}..."):
            detector = load_detector(config['model_name'])
        st.sidebar.success("✅ Modèle chargé")
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {e}")
        return
    
    # Zone principale
    st.markdown("---")
    
    # Chargement d'image
    image = render_image_upload()
    
    if image is None:
        st.info("👆 Chargez une image pour commencer la détection")
        render_footer()
        return
    
    # Convertir en RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Détection
    st.markdown("---")
    
    with st.spinner("🔍 Analyse en cours..."):
        image_np = image_to_array(image)
        detections = detector.detect(
            image_np,
            threshold=config['threshold'],
            max_detections=config['max_detections'],
            generate_approx_masks=config['generate_approx_masks']
        )
        
        # Filtrer par classe si nécessaire
        if config['selected_classes']:
            detections = [
                d for d in detections 
                if d.class_id in config['selected_classes']
            ]
    
    # Afficher les résultats
    if not detections:
        st.warning("⚠️ Aucun objet détecté. Essayez de réduire le seuil de confiance.")
        st.image(image, caption="Image originale", width="stretch")
    else:
        # Onglets pour différentes vues
        view_tab1, view_tab2 = st.tabs(["🎯 Résultat", "↔️ Comparaison"])
        
        with view_tab1:
            render_detection_results(image, detections, config)
        
        with view_tab2:
            render_comparison_view(image, detections, config)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
