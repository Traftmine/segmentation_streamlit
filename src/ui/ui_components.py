# -*- coding: utf-8 -*-
"""
Composants UI Streamlit réutilisables.
"""

import streamlit as st
import pathlib
from PIL import Image
from typing import List, Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import Detection
from core.constants import COCO_LABELS, AVAILABLE_MODELS
from core.detector import get_model_info
from utils.helpers import get_available_models
from utils.visualization import draw_detections, draw_masks_only


# =============================================================================
# COMPOSANTS SIDEBAR
# =============================================================================

def render_sidebar() -> Dict:
    """
    Affiche la barre latérale de configuration.
    
    Returns:
        Dictionnaire de configuration
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Section Modèle
        model_config = _render_model_selector()
        
        st.markdown("---")
        
        # Section Paramètres
        params_config = _render_parameters()
        
        st.markdown("---")
        
        # Section Affichage
        display_config = _render_display_options(model_config['model_name'])
        
        st.markdown("---")
        
        # Section Filtres
        filter_config = _render_class_filters()
        
        return {**model_config, **params_config, **display_config, **filter_config}


def _render_model_selector() -> Dict:
    """Affiche le sélecteur de modèle."""
    st.markdown("### 🤖 Modèle")
    models = get_available_models(AVAILABLE_MODELS)
    model_names = list(models.keys())
    
    # Grouper les modèles par famille
    model_families = {
        "SSD MobileNet": [m for m in model_names if "SSD" in m],
        "EfficientDet": [m for m in model_names if "EfficientDet" in m],
        "CenterNet": [m for m in model_names if "CenterNet" in m],
        "Faster R-CNN": [m for m in model_names if "Faster" in m],
        "Mask R-CNN 🎭": [m for m in model_names if "Mask" in m],
    }
    
    # Filtrer les familles vides
    model_families = {k: v for k, v in model_families.items() if v}
    
    # Sélection de la famille
    family = st.selectbox(
        "Famille de modèles",
        options=list(model_families.keys()),
        help="Choisissez une famille de modèles. 🎭 = supporte les masques natifs"
    )
    
    # Sélection du modèle dans la famille
    model_choice = st.selectbox(
        "Modèle",
        options=model_families[family],
        help="Choisissez un modèle spécifique"
    )
    
    # Afficher les infos du modèle
    model_info = get_model_info(model_choice)
    st.markdown(f"""
    <div class="model-info">
        <b>Vitesse:</b> {model_info.speed}<br>
        <b>Précision:</b> {model_info.accuracy}<br>
        <small>{model_info.description}</small>
    </div>
    """, unsafe_allow_html=True)
    
    return {'model_name': model_choice}


def _render_parameters() -> Dict:
    """Affiche les paramètres de détection."""
    st.markdown("### 🎚️ Paramètres")
    
    threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Score minimum pour afficher une détection"
    )
    
    max_detections = st.slider(
        "Nombre max de détections",
        min_value=1,
        max_value=100,
        value=20,
        help="Limite le nombre d'objets détectés"
    )
    
    return {
        'threshold': threshold,
        'max_detections': max_detections
    }


def _render_display_options(model_name: str) -> Dict:
    """Affiche les options d'affichage."""
    st.markdown("### 🎨 Affichage")
    
    show_boxes = st.checkbox("Boîtes englobantes", value=True)
    show_labels = st.checkbox("Labels", value=True)
    show_masks = st.checkbox("Masques de segmentation", value=True)
    
    # Options de masques
    generate_approx_masks = True
    mask_opacity = 100
    
    if show_masks:
        # Vérifier si le modèle supporte les masques natifs
        is_mask_model = "Mask" in model_name
        
        if not is_mask_model:
            st.info("💡 Ce modèle ne génère pas de masques natifs. Des masques elliptiques approximatifs seront utilisés.")
            generate_approx_masks = st.checkbox(
                "Activer masques approximatifs",
                value=True,
                help="Génère des masques elliptiques basés sur les boîtes"
            )
        else:
            st.success("🎭 Ce modèle génère des masques précis !")
            generate_approx_masks = False
        
        mask_opacity = st.slider(
            "Opacité des masques",
            min_value=0,
            max_value=255,
            value=100,
            help="Transparence des masques"
        )
    
    return {
        'show_boxes': show_boxes,
        'show_labels': show_labels,
        'show_masks': show_masks,
        'mask_opacity': mask_opacity,
        'generate_approx_masks': generate_approx_masks
    }


def _render_class_filters() -> Dict:
    """Affiche les filtres de classes."""
    st.markdown("### 🔍 Filtres de classes")
    
    filter_classes = st.checkbox("Filtrer par classe", value=False)
    selected_classes = None
    
    if filter_classes:
        # Sélection des classes
        class_options = list(COCO_LABELS.values())
        selected_class_names = st.multiselect(
            "Classes à détecter",
            options=class_options,
            default=["person", "car", "dog", "cat"]
        )
        # Convertir en IDs
        selected_classes = [
            k for k, v in COCO_LABELS.items() 
            if v in selected_class_names
        ]
    
    return {'selected_classes': selected_classes}


# =============================================================================
# COMPOSANTS PRINCIPAUX
# =============================================================================

def render_header():
    """Affiche l'en-tête de l'application."""
    st.markdown('<h1 class="main-title">🔍 Détection d\'Objets par IA</h1>', 
                unsafe_allow_html=True)
    st.markdown('''
    <p class="subtitle">
        Détectez et segmentez des objets dans vos images avec des modèles de deep learning
    </p>
    ''', unsafe_allow_html=True)


def render_image_upload() -> Optional[Image.Image]:
    """
    Affiche la zone de chargement d'image.
    
    Returns:
        Image PIL ou None si aucune image chargée
    """
    tab1, tab2 = st.tabs(["📤 Charger une image", "📁 Images d'exemple"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Glissez-déposez ou cliquez pour charger",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Formats: JPG, PNG, BMP, WEBP"
        )
        
        if uploaded_file:
            return Image.open(uploaded_file)
    
    with tab2:
        # Remonter au dossier racine du projet (JM_C) puis aller dans data/exemple
        data_path = pathlib.Path(__file__).parent.parent.parent / "data" / "exemple"
        
        # Lister les sous-dossiers (catégories d'animaux)
        categories = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        
        if categories:
            # Emoji pour chaque catégorie
            category_emojis = {
                'chat': '🐱', 'chien': '🐕', 'cheval': '🐴', 
                'vache': '🐄', 'mouton': '🐑', 'coq': '🐓', 
                'écureil': '🐿️', 'ecureuil': '🐿️'
            }
            
            # Sélecteur de catégorie
            selected_category = st.selectbox(
                "🏷️ Choisir une catégorie",
                options=categories,
                format_func=lambda x: f"{category_emojis.get(x.lower(), '🐾')} {x.capitalize()}"
            )
            
            # Charger les images de la catégorie sélectionnée
            category_path = data_path / selected_category
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            sample_images = []
            for ext in extensions:
                sample_images.extend(category_path.glob(ext))
            
            if sample_images:
                # Limiter à 8 images max et afficher en grille
                num_images = min(len(sample_images), 8)
                st.caption(f"{len(sample_images)} image(s) disponible(s)")
                
                cols = st.columns(4)
                for idx, img_path in enumerate(sample_images[:num_images]):
                    with cols[idx % 4]:
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.stem[:15], use_container_width=True)
                        if st.button("Utiliser", key=f"sample_{selected_category}_{idx}"):
                            return img
            else:
                st.info(f"Aucune image dans le dossier {selected_category}/")
        else:
            st.info("Aucun sous-dossier d'images trouvé dans data/exemple/")
    
    return None


def render_stats(detections: List[Detection]):
    """Affiche les statistiques de détection."""
    if not detections:
        return
    
    # Compter par classe
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(detections)}</div>
            <div class="stat-label">Objets détectés</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(class_counts)}</div>
            <div class="stat-label">Classes uniques</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_conf = sum(d.confidence for d in detections) / len(detections)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_conf:.0%}</div>
            <div class="stat-label">Confiance moyenne</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        masks_count = sum(1 for d in detections if d.mask is not None)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{masks_count}</div>
            <div class="stat-label">Masques générés</div>
        </div>
        """, unsafe_allow_html=True)


def render_detection_results(
    image: Image.Image,
    detections: List[Detection],
    config: dict
):
    """Affiche les résultats de détection."""
    
    # Dessiner les détections
    result_image = draw_detections(
        image,
        detections,
        show_boxes=config['show_boxes'],
        show_labels=config['show_labels'],
        show_masks=config['show_masks'],
        mask_alpha=config['mask_opacity']
    )
    
    # Afficher l'image résultat
    st.image(result_image, caption="Résultat de la détection", 
             width="stretch")
    
    # Afficher les statistiques
    render_stats(detections)
    
    # Liste des détections
    if detections:
        st.markdown("### 📋 Détails des détections")
        
        for i, det in enumerate(detections, 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                mask_icon = "🎭" if det.mask is not None else ""
                st.markdown(f"**{i}. {det.class_name}** {mask_icon}")
            
            with col2:
                st.progress(det.confidence)
            
            with col3:
                st.write(f"{det.confidence:.1%}")


def render_comparison_view(
    image: Image.Image,
    detections: List[Detection],
    config: dict
):
    """Affiche une vue comparée original/détections."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 Image originale")
        st.image(image, width="stretch")
    
    with col2:
        st.markdown("#### 🎯 Détections")
        result_image = draw_detections(
            image,
            detections,
            show_boxes=config['show_boxes'],
            show_labels=config['show_labels'],
            show_masks=config['show_masks'],
            mask_alpha=config['mask_opacity']
        )
        st.image(result_image, width="stretch")
    
    # Vue masques uniquement si disponible
    has_masks = any(d.mask is not None for d in detections)
    if has_masks and config['show_masks']:
        st.markdown("#### 🎭 Vue segmentation")
        mask_image = draw_masks_only(image, detections, alpha=180)
        st.image(mask_image, width="stretch")


def render_footer():
    """Affiche le pied de page."""
    st.markdown("""
    <div class="footer">
        <p>🚀 Propulsé par <b>TensorFlow Hub</b> et <b>Streamlit</b></p>
        <p>Modèles pré-entraînés sur COCO Dataset (80 classes d'objets)</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">
            SSD MobileNet • EfficientDet • CenterNet • Faster R-CNN • Mask R-CNN 🎭
        </p>
    </div>
    """, unsafe_allow_html=True)
