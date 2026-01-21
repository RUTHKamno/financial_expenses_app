import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TensorFlowPredictor:
    """
    Classe pour charger et utiliser un modèle TensorFlow pour la prédiction des dépenses bancaires.
    """
    
    def __init__(self, model_path='models/best_model.h5', scaler_path='models/scaler.pkl'):
        """
        Initialise le prédicteur TensorFlow.
        
        Args:
            model_path (str): Chemin vers le modèle TensorFlow sauvegardé
            scaler_path (str): Chemin vers le scaler sauvegardé
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
        # Charger le modèle et le scaler
        self.load_model()
        self.load_scaler()
    
    def load_model(self):
        """
        Charge le modèle TensorFlow depuis le fichier.
        Si le modèle n'existe pas, crée un modèle par défaut.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"✅ Modèle TensorFlow chargé depuis {self.model_path}")
            else:
                print(f"⚠️ Modèle non trouvé. Création d'un modèle par défaut...")
                self.model = self.create_default_model()
                # Créer le dossier si nécessaire
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                print(f"✅ Modèle par défaut créé et sauvegardé")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            print("Création d'un modèle par défaut...")
            self.model = self.create_default_model()
    
    def create_default_model(self):
        """
        Crée un modèle TensorFlow par défaut pour la prédiction des dépenses.
        
        Returns:
            keras.Model: Modèle TensorFlow compilé
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(9,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')  # Régression
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def load_scaler(self):
        """
        Charge le scaler depuis le fichier.
        Si le scaler n'existe pas, crée un scaler par défaut.
        """
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Scaler chargé depuis {self.scaler_path}")
            else:
                print(f"⚠️ Scaler non trouvé. Création d'un scaler par défaut...")
                self.scaler = StandardScaler()
                # Fit avec des données factices pour initialisation
                dummy_data = np.array([[50000, 25000, 2500, 5, 8000, 40000, 0.3, 0.25, 0.27]])
                self.scaler.fit(dummy_data)
                # Créer le dossier si nécessaire
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                joblib.dump(self.scaler, self.scaler_path)
                print(f"✅ Scaler par défaut créé et sauvegardé")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du scaler: {str(e)}")
            print("Création d'un scaler par défaut...")
            self.scaler = StandardScaler()
            dummy_data = np.array([[50000, 25000, 2500, 5, 8000, 40000, 0.3, 0.25, 0.27]])
            self.scaler.fit(dummy_data)
    
    def preprocess(self, X):
        """
        Prétraite les données avant la prédiction.
        
        Args:
            X (np.ndarray): Données d'entrée (n_samples, 9)
        
        Returns:
            np.ndarray: Données prétraitées
        """
        # Vérifier la forme des données
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Standardisation
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, X):
        """
        Effectue une prédiction sur les données.
        
        Args:
            X (np.ndarray): Données d'entrée (n_samples, 9)
                Colonnes: [bilan_financier, actifs, revenu, taux_interet, 
                          flux_tresorerie, capital, agence_freq, banque_freq, lieu_freq]
        
        Returns:
            np.ndarray: Prédictions des dépenses
        """
        # Prétraitement
        X_processed = self.preprocess(X)
        
        # Prédiction
        predictions = self.model.predict(X_processed, verbose=0)
        
        # Aplatir les résultats
        predictions = predictions.flatten()
        
        # S'assurer que les prédictions sont positives
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_confidence(self, X, n_iterations=100):
        """
        Effectue une prédiction avec intervalle de confiance (Monte Carlo Dropout).
        
        Args:
            X (np.ndarray): Données d'entrée
            n_iterations (int): Nombre d'itérations pour l'estimation
        
        Returns:
            tuple: (prédiction moyenne, écart-type, intervalle de confiance)
        """
        # Prétraitement
        X_processed = self.preprocess(X)
        
        # Prédictions multiples avec dropout activé
        predictions = []
        for _ in range(n_iterations):
            pred = self.model(X_processed, training=True)
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        # Calcul des statistiques
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Intervalle de confiance à 95%
        confidence_interval = (
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred
        )
        
        return mean_pred, std_pred, confidence_interval
    
    def get_model_summary(self):
        """
        Retourne un résumé du modèle.
        
        Returns:
            str: Résumé du modèle
        """
        if self.model is not None:
            from io import StringIO
            stream = StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "Modèle non chargé"
    
    def evaluate(self, X, y):
        """
        Évalue le modèle sur des données de test.
        
        Args:
            X (np.ndarray): Données d'entrée
            y (np.ndarray): Valeurs réelles
        
        Returns:
            dict: Métriques d'évaluation
        """
        X_processed = self.preprocess(X)
        
        # Évaluation
        results = self.model.evaluate(X_processed, y, verbose=0)
        
        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2]
        }
        
        return metrics