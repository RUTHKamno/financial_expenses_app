"""
Module utilitaire pour l'application de prédiction des dépenses bancaires.

Ce module contient les classes et fonctions nécessaires pour:
- Charger et utiliser les modèles TensorFlow et PyTorch
- Prétraiter les données
- Effectuer des prédictions
- Évaluer les performances des modèles
"""

from .tensorflow_model import TensorFlowPredictor
# from .pytorch_model import PyTorchPredictor, BankingNet

__all__ = [
    'TensorFlowPredictor',
    # 'PyTorchPredictor',
    # 'BankingNet'
]

__version__ = '1.0.0'