# -*- coding: utf-8 -*-
"""
Module ui - Interface utilisateur Streamlit.

Contient:
- styles.py        : Styles CSS
- ui_components.py : Composants UI
"""

from .styles import (
    APP_CSS,
    inject_css
)

from .ui_components import (
    render_sidebar,
    render_header,
    render_image_upload,
    render_stats,
    render_detection_results,
    render_comparison_view,
    render_footer
)

__all__ = [
    'APP_CSS',
    'inject_css',
    'render_sidebar',
    'render_header',
    'render_image_upload',
    'render_stats',
    'render_detection_results',
    'render_comparison_view',
    'render_footer',
]
