# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# import joblib
# import os

# class BankingNet(nn.Module):
#     """
#     Réseau de neurones PyTorch pour la prédiction des dépenses bancaires.
#     """
    
#     def __init__(self, input_size=9, hidden_sizes=[32, 16, 8], output_size=1):
#         """
#         Initialise le réseau de neurones.
        
#         Args:
#             input_size (int): Nombre de features d'entrée
#             hidden_sizes (list): Liste des tailles des couches cachées
#             output_size (int): Nombre de sorties (1 pour la régression)
#         """
#         super(BankingNet, self).__init__()
        
#         # Couches du réseau
#         layers = []
#         prev_size = input_size
        
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.2)
#             ])
#             prev_size = hidden_size
        
#         # Couche de sortie
#         layers.append(nn.Linear(prev_size, output_size))
        
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         """
#         Passage avant dans le réseau.
        
#         Args:
#             x (torch.Tensor): Données d'entrée
        
#         Returns:
#             torch.Tensor: Prédictions
#         """
#         return self.network(x)


# class PyTorchPredictor:
#     """
#     Classe pour charger et utiliser un modèle PyTorch pour la prédiction des dépenses bancaires.
#     """
    
#     def __init__(self, model_path='models/best_model.pth', scaler_path='models/scaler.pkl'):
#         """
#         Initialise le prédicteur PyTorch.
        
#         Args:
#             model_path (str): Chemin vers le modèle PyTorch sauvegardé
#             scaler_path (str): Chemin vers le scaler sauvegardé
#         """
#         self.model_path = model_path
#         self.scaler_path = scaler_path
#         self.model = None
#         self.scaler = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Charger le modèle et le scaler
#         self.load_model()
#         self.load_scaler()
    
#     def load_model(self):
#         """
#         Charge le modèle PyTorch depuis le fichier.
#         Si le modèle n'existe pas, crée un modèle par défaut.
#         """
#         try:
#             self.model = BankingNet()
            
#             if os.path.exists(self.model_path):
#                 # Charger les poids du modèle
#                 checkpoint = torch.load(self.model_path, map_location=self.device)
                
#                 # Support pour différents formats de sauvegarde
#                 if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                     self.model.load_state_dict(checkpoint['model_state_dict'])
#                 else:
#                     self.model.load_state_dict(checkpoint)
                
#                 self.model.to(self.device)
#                 self.model.eval()
#                 print(f"✅ Modèle PyTorch chargé depuis {self.model_path}")
#             else:
#                 print(f"⚠️ Modèle non trouvé. Création d'un modèle par défaut...")
#                 self.model.to(self.device)
#                 # Créer le dossier si nécessaire
#                 os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#                 torch.save(self.model.state_dict(), self.model_path)
#                 print(f"✅ Modèle par défaut créé et sauvegardé")
#         except Exception as e:
#             print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
#             print("Création d'un modèle par défaut...")
#             self.model = BankingNet()
#             self.model.to(self.device)
    
#     def load_scaler(self):
#         """
#         Charge le scaler depuis le fichier.
#         Si le scaler n'existe pas, crée un scaler par défaut.
#         """
#         try:
#             if os.path.exists(self.scaler_path):
#                 self.scaler = joblib.load(self.scaler_path)
#                 print(f"✅ Scaler chargé depuis {self.scaler_path}")
#             else:
#                 print(f"⚠️ Scaler non trouvé. Création d'un scaler par défaut...")
#                 self.scaler = StandardScaler()
#                 # Fit avec des données factices pour initialisation
#                 dummy_data = np.array([[50000, 25000, 2500, 5, 8000, 40000, 0.3, 0.25, 0.27]])
#                 self.scaler.fit(dummy_data)
#                 # Créer le dossier si nécessaire
#                 os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
#                 joblib.dump(self.scaler, self.scaler_path)
#                 print(f"✅ Scaler par défaut créé et sauvegardé")
#         except Exception as e:
#             print(f"❌ Erreur lors du chargement du scaler: {str(e)}")
#             print("Création d'un scaler par défaut...")
#             self.scaler = StandardScaler()
#             dummy_data = np.array([[50000, 25000, 2500, 5, 8000, 40000, 0.3, 0.25, 0.27]])
#             self.scaler.fit(dummy_data)
    
#     def preprocess(self, X):
#         """
#         Prétraite les données avant la prédiction.
        
#         Args:
#             X (np.ndarray): Données d'entrée (n_samples, 9)
        
#         Returns:
#             torch.Tensor: Données prétraitées sous forme de tensor
#         """
#         # Vérifier la forme des données
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
        
#         # Standardisation
#         X_scaled = self.scaler.transform(X)
        
#         # Conversion en tensor PyTorch
#         X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        
#         return X_tensor
    
#     def predict(self, X):
#         """
#         Effectue une prédiction sur les données.
        
#         Args:
#             X (np.ndarray): Données d'entrée (n_samples, 9)
#                 Colonnes: [bilan_financier, actifs, revenu, taux_interet, 
#                           flux_tresorerie, capital, agence_freq, banque_freq, lieu_freq]
        
#         Returns:
#             np.ndarray: Prédictions des dépenses
#         """
#         self.model.eval()
        
#         with torch.no_grad():
#             # Prétraitement
#             X_tensor = self.preprocess(X)
            
#             # Prédiction
#             predictions = self.model(X_tensor)
            
#             # Conversion en numpy
#             predictions = predictions.cpu().numpy().flatten()
            
#             # S'assurer que les prédictions sont positives
#             predictions = np.maximum(predictions, 0)
        
#         return predictions
    
#     def predict_with_confidence(self, X, n_iterations=100):
#         """
#         Effectue une prédiction avec intervalle de confiance (Monte Carlo Dropout).
        
#         Args:
#             X (np.ndarray): Données d'entrée
#             n_iterations (int): Nombre d'itérations pour l'estimation
        
#         Returns:
#             tuple: (prédiction moyenne, écart-type, intervalle de confiance)
#         """
#         self.model.train()  # Activer le dropout
        
#         # Prétraitement
#         X_tensor = self.preprocess(X)
        
#         # Prédictions multiples avec dropout activé
#         predictions = []
#         with torch.no_grad():
#             for _ in range(n_iterations):
#                 pred = self.model(X_tensor)
#                 predictions.append(pred.cpu().numpy().flatten())
        
#         predictions = np.array(predictions)
        
#         # Calcul des statistiques
#         mean_pred = np.mean(predictions, axis=0)
#         std_pred = np.std(predictions, axis=0)
        
#         # Intervalle de confiance à 95%
#         confidence_interval = (
#             mean_pred - 1.96 * std_pred,
#             mean_pred + 1.96 * std_pred
#         )
        
#         self.model.eval()  # Retour en mode évaluation
        
#         return mean_pred, std_pred, confidence_interval
    
#     def get_model_summary(self):
#         """
#         Retourne un résumé du modèle.
        
#         Returns:
#             str: Résumé du modèle
#         """
#         summary = []
#         summary.append("=" * 60)
#         summary.append("PyTorch Banking Prediction Model")
#         summary.append("=" * 60)
        
#         total_params = 0
#         trainable_params = 0
        
#         for name, param in self.model.named_parameters():
#             params = param.numel()
#             total_params += params
#             if param.requires_grad:
#                 trainable_params += params
#             summary.append(f"{name:40s} {str(param.shape):20s} {params:>10,d} params")
        
#         summary.append("=" * 60)
#         summary.append(f"Total parameters: {total_params:,}")
#         summary.append(f"Trainable parameters: {trainable_params:,}")
#         summary.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
#         summary.append(f"Device: {self.device}")
#         summary.append("=" * 60)
        
#         return "\n".join(summary)
    
#     def evaluate(self, X, y):
#         """
#         Évalue le modèle sur des données de test.
        
#         Args:
#             X (np.ndarray): Données d'entrée
#             y (np.ndarray): Valeurs réelles
        
#         Returns:
#             dict: Métriques d'évaluation
#         """
#         self.model.eval()
        
#         with torch.no_grad():
#             # Préparation des données
#             X_tensor = self.preprocess(X)
#             y_tensor = torch.FloatTensor(y).to(self.device)
            
#             # Prédictions
#             predictions = self.model(X_tensor).squeeze()
            
#             # Calcul des métriques
#             mse = nn.MSELoss()(predictions, y_tensor).item()
#             mae = nn.L1Loss()(predictions, y_tensor).item()
            
#             # Calcul du R²
#             y_mean = torch.mean(y_tensor)
#             ss_tot = torch.sum((y_tensor - y_mean) ** 2)
#             ss_res = torch.sum((y_tensor - predictions) ** 2)
#             r2 = 1 - (ss_res / ss_tot).item()
        
#         metrics = {
#             'mse': mse,
#             'mae': mae,
#             'rmse': np.sqrt(mse),
#             'r2': r2
#         }
        
#         return metrics
    
#     def save_model(self, path=None):
#         """
#         Sauvegarde le modèle.
        
#         Args:
#             path (str): Chemin de sauvegarde (utilise self.model_path par défaut)
#         """
#         if path is None:
#             path = self.model_path
        
#         os.makedirs(os.path.dirname(path), exist_ok=True)
        
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'model_architecture': str(self.model)
#         }
        
#         torch.save(checkpoint, path)
#         print(f"✅ Modèle sauvegardé dans {path}")