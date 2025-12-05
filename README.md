# 🔍 JM_C - Détection d'Objets avec TensorFlow

Application Streamlit de détection d'objets utilisant des modèles pré-entraînés TensorFlow Hub sur le dataset COCO.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Fonctionnalités

- **Détection d'objets** : Détecte 80 classes d'objets du dataset COCO
- **Segmentation d'instance** : Support des masques avec Mask R-CNN
- **Multiple modèles** : 14 modèles disponibles (SSD, EfficientDet, CenterNet, Faster R-CNN, Mask R-CNN)
- **Interface interactive** : Application web Streamlit intuitive
- **Images d'exemple** : Galerie d'images classées par catégorie d'animaux

## 🏗️ Architecture du projet

```
JM_C/
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation
├── data/
│   └── exemple/              # Images d'exemple
│       ├── chat/
│       ├── chien/
│       ├── cheval/
│       └── ...
└── src/
    ├── app.py                # Point d'entrée Streamlit
    ├── config.py             # Configuration globale
    │
    ├── core/                 # Logique métier
    │   ├── constants.py      # Labels COCO, modèles disponibles
    │   ├── data_types.py     # Detection, ModelInfo (dataclasses)
    │   └── detector.py       # ObjectDetector
    │
    ├── ui/                   # Interface utilisateur
    │   ├── styles.py         # CSS personnalisé
    │   └── ui_components.py  # Composants Streamlit
    │
    ├── utils/                # Utilitaires
    │   ├── colors.py         # Gestion des couleurs
    │   ├── helpers.py        # Fonctions utilitaires
    │   ├── image_utils.py    # Manipulation d'images
    │   └── visualization.py  # Dessin des détections
    │
    └── tests/                # Tests unitaires
        ├── conftest.py
        ├── test_colors.py
        ├── test_constants.py
        ├── test_data_types.py
        ├── test_detector.py
        ├── test_helpers.py
        └── test_image_utils.py
```

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

### Étapes

1. **Cloner le repository**
   ```bash
   git clone <url-du-repo>
   cd JM_C
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # ou
   .venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Utilisation

### Lancer l'application

```bash
cd src
streamlit run app.py
```

L'application sera accessible à l'adresse : http://localhost:8501

### Interface

1. **Sidebar** : Sélection du modèle, seuil de confiance, options d'affichage
2. **Zone principale** :
   - Onglet "Charger une image" : Upload de vos propres images
   - Onglet "Images d'exemple" : Galerie par catégorie d'animaux
3. **Résultats** : Visualisation des détections avec boîtes englobantes et masques

## 🤖 Modèles disponibles

| Modèle | Type | Vitesse | Précision |
|--------|------|---------|-----------|
| SSD MobileNet V2 | Détection | ⚡ Très rapide | ★★☆☆☆ |
| SSD MobileNet V2 FPNLite 320 | Détection | ⚡ Très rapide | ★★★☆☆ |
| SSD MobileNet V2 FPNLite 640 | Détection | 🚀 Rapide | ★★★★☆ |
| EfficientDet D0-D3 | Détection | 🚀-🐢 Variable | ★★★-★★★★★ |
| CenterNet HourGlass104 | Détection | 🐢 Lent | ★★★★★ |
| Faster R-CNN ResNet | Détection | 🐢 Lent | ★★★★★ |
| Mask R-CNN Inception ResNet V2 | Segmentation | 🐢 Très lent | ★★★★★ |

## 🧪 Tests

Exécuter les tests unitaires :

```bash
cd src
python -m pytest tests/ -v
```

Avec couverture de code :

```bash
python -m pytest tests/ -v --cov=. --cov-report=html
```

## 📊 Classes COCO détectables

L'application peut détecter 80 classes d'objets, notamment :

- **Personnes** : person
- **Véhicules** : bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animaux** : bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objets du quotidien** : backpack, umbrella, handbag, suitcase, bottle, cup, fork, knife, spoon, bowl
- **Nourriture** : banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Mobilier** : chair, couch, bed, dining table, toilet, tv, laptop
- **Électronique** : mouse, remote, keyboard, cell phone, microwave, oven, toaster, refrigerator

## 🔧 Configuration

Les paramètres de l'application peuvent être modifiés dans `src/config.py` :

- Seuil de confiance par défaut
- Nombre maximum de détections
- Options d'affichage

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- Projet réalisé dans le cadre du Master 2 MIX

## 🙏 Remerciements

- [TensorFlow Hub](https://tfhub.dev/) pour les modèles pré-entraînés
- [COCO Dataset](https://cocodataset.org/) pour les données d'entraînement
- [Streamlit](https://streamlit.io/) pour le framework d'application web
