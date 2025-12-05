# -*- coding: utf-8 -*-
"""
Styles CSS pour l'application Streamlit.
"""

APP_CSS = """
<style>
    /* Style général */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Titre principal */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Sous-titre */
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Cartes de stats */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Info modèle */
    .model-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Badge de détection */
    .detection-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""


def inject_css():
    """Injecte le CSS personnalisé dans Streamlit."""
    import streamlit as st
    st.markdown(APP_CSS, unsafe_allow_html=True)
