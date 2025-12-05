# -*- coding: utf-8 -*-
"""
Constantes de l'application.
Labels COCO et définitions des modèles disponibles.
"""

from typing import Dict


# =============================================================================
# LABELS COCO (80 CLASSES)
# =============================================================================

COCO_LABELS: Dict[int, str] = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}


# =============================================================================
# MODÈLES DISPONIBLES SUR TENSORFLOW HUB
# =============================================================================

AVAILABLE_MODELS: Dict[str, Dict] = {
    # Modèles de détection rapides
    "SSD MobileNet V2": {
        "url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        "type": "detection",
        "speed": "⚡ Très rapide",
        "accuracy": "★★☆☆☆",
        "description": "Idéal pour le temps réel, moins précis"
    },
    "SSD MobileNet V2 FPNLite 320": {
        "url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
        "type": "detection",
        "speed": "⚡ Très rapide",
        "accuracy": "★★★☆☆",
        "description": "Version améliorée avec FPN"
    },
    "SSD MobileNet V2 FPNLite 640": {
        "url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1",
        "type": "detection",
        "speed": "🚀 Rapide",
        "accuracy": "★★★★☆",
        "description": "Haute résolution, meilleure précision"
    },
    # Modèles EfficientDet
    "EfficientDet D0": {
        "url": "https://tfhub.dev/tensorflow/efficientdet/d0/1",
        "type": "detection",
        "speed": "🚀 Rapide",
        "accuracy": "★★★☆☆",
        "description": "Bon équilibre vitesse/précision"
    },
    "EfficientDet D1": {
        "url": "https://tfhub.dev/tensorflow/efficientdet/d1/1",
        "type": "detection",
        "speed": "🔄 Modéré",
        "accuracy": "★★★★☆",
        "description": "Plus précis que D0"
    },
    "EfficientDet D2": {
        "url": "https://tfhub.dev/tensorflow/efficientdet/d2/1",
        "type": "detection",
        "speed": "🔄 Modéré",
        "accuracy": "★★★★☆",
        "description": "Encore plus précis"
    },
    "EfficientDet D3": {
        "url": "https://tfhub.dev/tensorflow/efficientdet/d3/1",
        "type": "detection",
        "speed": "🐢 Lent",
        "accuracy": "★★★★★",
        "description": "Très précis, plus lent"
    },
    # Modèles CenterNet
    "CenterNet HourGlass104": {
        "url": "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1",
        "type": "detection",
        "speed": "🐢 Lent",
        "accuracy": "★★★★★",
        "description": "Excellente précision pour objets centrés"
    },
    "CenterNet Resnet50 V1 FPN": {
        "url": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1",
        "type": "detection",
        "speed": "🔄 Modéré",
        "accuracy": "★★★★☆",
        "description": "Bon compromis avec ResNet backbone"
    },
    # Faster R-CNN
    "Faster R-CNN ResNet50 V1": {
        "url": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
        "type": "detection",
        "speed": "🐢 Lent",
        "accuracy": "★★★★★",
        "description": "Très précis, classique"
    },
    "Faster R-CNN ResNet101 V1": {
        "url": "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1",
        "type": "detection",
        "speed": "🐢 Très lent",
        "accuracy": "★★★★★",
        "description": "Maximum de précision"
    },
    "Faster R-CNN Inception ResNet V2": {
        "url": "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1",
        "type": "detection",
        "speed": "🐢 Très lent",
        "accuracy": "★★★★★",
        "description": "Backbone très puissant"
    },
    # Modèles avec segmentation (Mask R-CNN)
    "Mask R-CNN Inception ResNet V2": {
        "url": "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1",
        "type": "segmentation",
        "speed": "🐢 Très lent",
        "accuracy": "★★★★★",
        "description": "Segmentation d'instance - génère des masques précis",
        "supports_masks": True
    },
}
